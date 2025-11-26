import streamlit as st
import os
import json
import base64
import io
import re
import copy
import asyncio
import socket
import time
import html as html_lib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from openai import AsyncOpenAI
from googleapiclient.errors import HttpError

# --- タイムアウト設定 ---
socket.setdefaulttimeout(300)

# --- ページ設定 ---
st.set_page_config(
    page_title="画像チェックAIツール テスト用",
    layout="wide"
)

# --- CSS読み込み ---
def local_css(file_name):
    try:
        with open(file_name, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass
    except UnicodeDecodeError as e:
        st.error(f"CSSファイルの読み込みエラー: {e}")

local_css("style.css")

# ==========================================
# ログイン機能 (Google認証)
# ==========================================
if not st.user.get("is_logged_in", False):
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 30px;'>
            画像チェックAIツール
        </h1>
    """, unsafe_allow_html=True)

    _, form_col, _ = st.columns([3, 2, 3])
    with form_col:
        st.markdown(f'Googleアカウントでログインしてください。')
        if st.button("Googleアカウントでログイン", icon=":material/login:", width='stretch'):
            st.login() 

    st.stop()

# ==========================================
# メインアプリケーション
# ==========================================
st.markdown("""
    <style>
    [data-testid="column"]:nth-of-type(2) {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        justify-content: flex-end;
    }
    </style>
""", unsafe_allow_html=True)

header_col1, header_col2 = st.columns([8, 2], vertical_alignment="bottom")

with header_col1:
    st.title("画像チェックAIツール テスト用")
    st.markdown("1プロンプト+複数画像のリクエストを**OpenAI Batch API**で処理します。")

with header_col2:
    st.markdown(
        f"<div style='text-align: right; font-size: 0.85rem; margin-bottom: 4px; color: #666;'>User: <b>{st.user.name}</b></div>", 
        unsafe_allow_html=True
    )
    if st.button("ログアウト", icon=":material/logout:", key="logout_btn", use_container_width=True):
        st.logout()

st.divider()

# --- 定数 ---
IMAGES_PER_GROUP = 10 
MAX_DL_WORKERS = 20
MAX_RETRIES = 3

# --- OpenAIクライアント (非同期) ---
if "openai" in st.secrets:
    client = AsyncOpenAI(api_key=st.secrets["openai"]["api_key"])
else:
    st.error("SecretsにOpenAI APIキーが設定されていません。")
    st.stop()

# --- セッションステート初期化 ---
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = None
if "prompt_logs" not in st.session_state:
    st.session_state.prompt_logs = None

# --- 関数定義 ---

def get_credentials():
    if "google" not in st.secrets or "credentials_json" not in st.secrets["google"]:
        st.error("SecretsにGoogle認証情報が設定されていません。")
        st.stop()
    try:
        json_str = st.secrets["google"]["credentials_json"]
        gcp_info = json.loads(json_str)
        return service_account.Credentials.from_service_account_info(
            gcp_info, scopes=['https://www.googleapis.com/auth/drive']
        )
    except Exception as e:
        st.error(f"認証情報の読み込みエラー: {e}")
        st.stop()

def extract_folder_id(url_or_id):
    match = re.search(r'folders/([a-zA-Z0-9_-]+)', url_or_id)
    return match.group(1) if match else url_or_id

def clean_prompt_for_display(messages):
    """表示用にBase64を省略する関数"""
    cleaned = copy.deepcopy(messages)
    for msg in cleaned:
        if msg.get('role') == 'user' and isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'image_url':
                    item['image_url']['url'] = "【画像データ(Base64)は省略しています】"
    return cleaned

def download_image_thread_safe(credentials, file_id):
    for attempt in range(MAX_RETRIES):
        try:
            local_service = build('drive', 'v3', credentials=credentials)
            request = local_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            return fh.read()
        except (socket.timeout, HttpError) as e:
            time.sleep(1 * (attempt + 1))
            continue
        except Exception:
            return None
    return None

async def download_images_parallel(credentials, files):
    loop = asyncio.get_running_loop()
    total = len(files)
    completed = 0
    
    status_text = st.empty()
    progress_bar = st.progress(0)

    with ThreadPoolExecutor(max_workers=MAX_DL_WORKERS) as executor:
        tasks = []
        for f in files:
            task = loop.run_in_executor(executor, download_image_thread_safe, credentials, f['id'])
            tasks.append((f, task))
        
        async def run_with_progress(file_info, future):
            data = await future
            nonlocal completed
            completed += 1
            progress_bar.progress(completed / total)
            status_text.write(f"画像ダウンロード中... {completed}/{total}")
            
            if data:
                b64 = base64.b64encode(data).decode('utf-8')
                return {"name": file_info['name'], "base64": b64}
            return None

        coroutines = [run_with_progress(f, t) for f, t in tasks]
        results = await asyncio.gather(*coroutines)
    
    status_text.empty()
    progress_bar.empty()
    return [r for r in results if r is not None]

def create_message_for_group(image_group):
    """1グループ分のメッセージリストを作成するヘルパー関数"""
    content_list = [
        {
            "type": "text", 
            "text": (
                "以下の複数の画像をOCRしてください。\n"
                "出力は必ず以下のJSON形式のみを返してください。Markdown記法は不要です。\n"
                "形式: {\"ファイル名A.jpg\": \"抽出テキストA\", \"ファイル名B.jpg\": \"抽出テキストB\"...}"
            )
        }
    ]
    
    file_names_in_group = []
    for img in image_group:
        file_names_in_group.append(img['name'])
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img['base64']}",
                "detail": "high"
            }
        })
        content_list.append({
            "type": "text", 
            "text": f"Filename: {img['name']}"
        })

    messages = [
        {"role": "system", "content": "You are an OCR assistant."},
        {"role": "user", "content": content_list}
    ]
    return messages, file_names_in_group

async def run_batch_api_process(images):
    """Batch APIを使用して処理を実行する"""
    
    # 1. 画像をグループ化
    groups = [images[i:i + IMAGES_PER_GROUP] for i in range(0, len(images), IMAGES_PER_GROUP)]
    total_groups = len(groups)
    
    st.toast(f"{total_groups}個のバッチリクエストを作成中...", icon="📦")
    
    # 2. JSONLデータの作成 (メモリ上)
    jsonl_buffer = io.BytesIO()
    prompt_logs = []
    
    for i, group in enumerate(groups):
        group_id = i + 1
        messages, file_names = create_message_for_group(group)
        
        # ログ保存用
        display_log = clean_prompt_for_display(messages)
        prompt_logs.append({
            "group_id": group_id,
            "files": file_names,
            "messages": display_log
        })
        
        # Batch Request オブジェクト
        request_obj = {
            "custom_id": f"group_{group_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 4000,
                "temperature": 0.0
            }
        }
        
        # JSONLとして書き込み
        json_line = json.dumps(request_obj) + "\n"
        jsonl_buffer.write(json_line.encode('utf-8'))
    
    # バッファのポインタを先頭へ
    jsonl_buffer.seek(0)
    
    # 3. ファイルアップロード
    st.toast("OpenAIにバッチファイルをアップロード中...", icon="☁️")
    try:
        batch_input_file = await client.files.create(
            file=jsonl_buffer,
            purpose="batch"
        )
    except Exception as e:
        st.error(f"ファイルアップロードエラー: {e}")
        return [], prompt_logs

    # 4. バッチ作成
    st.toast("バッチジョブを作成中...", icon="🚀")
    try:
        batch_job = await client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h" # 24時間以内に完了
        )
    except Exception as e:
        st.error(f"バッチ作成エラー: {e}")
        return [], prompt_logs

    # 5. ポーリング (待機)
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    start_time = time.time()
    
    while True:
        try:
            batch_status = await client.batches.retrieve(batch_job.id)
            status = batch_status.status
            
            # 進捗表示更新
            elapsed = int(time.time() - start_time)
            status_msg = f"Batch Status: **{status}** (経過時間: {elapsed}秒)"
            
            if batch_status.request_counts:
                total_req = batch_status.request_counts.total or total_groups
                completed_req = batch_status.request_counts.completed or 0
                if total_req > 0:
                    progress_bar.progress(completed_req / total_req)
                    status_msg += f" - 進捗: {completed_req}/{total_req}"

            status_placeholder.markdown(f'<div class="blinking-text">{status_msg}</div>', unsafe_allow_html=True)

            if status == 'completed':
                progress_bar.progress(1.0)
                st.success("バッチ処理が完了しました！結果をダウンロードします。")
                break
            elif status in ['failed', 'cancelled', 'expired']:
                st.error(f"バッチ処理が失敗またはキャンセルされました。Status: {status}")
                # エラー詳細があれば表示
                if batch_status.errors:
                    st.json(batch_status.errors)
                return [], prompt_logs
            
            # 待機 (Batch APIは時間がかかるため、少し長めに待機)
            await asyncio.sleep(5) 
            
        except Exception as e:
            st.error(f"ステータス確認中にエラー: {e}")
            await asyncio.sleep(5)

    status_placeholder.empty()
    progress_bar.empty()

    # 6. 結果取得
    if batch_status.output_file_id:
        try:
            file_response = await client.files.content(batch_status.output_file_id)
            result_content = file_response.text
            
            # 結果のパース
            final_ocr_results = []
            
            # custom_id (group_X) と 画像グループのマッピングが必要
            # 簡易的に、outputのjsonlをパースして処理
            
            for line in result_content.strip().split('\n'):
                if not line: continue
                res_json = json.loads(line)
                custom_id = res_json.get('custom_id') # group_1
                
                # グループIDから元の画像情報を特定する処理
                # group_id は 1始まりのインデックス
                g_idx = int(custom_id.replace("group_", "")) - 1
                original_group = groups[g_idx]
                
                # APIレスポンス内容
                response_body = res_json.get('response', {}).get('body', {})
                choices = response_body.get('choices', [])
                
                if choices:
                    content_text = choices[0].get('message', {}).get('content', "")
                    cleaned_json_str = re.sub(r"```json\n?|```", "", content_text).strip()
                    try:
                        parsed_ocr_json = json.loads(cleaned_json_str)
                    except:
                        parsed_ocr_json = {}
                        
                    # 結果を整形
                    for img in original_group:
                        fname = img['name']
                        ocr_text = parsed_ocr_json.get(fname, "（抽出失敗または順序エラー）")
                        final_ocr_results.append({
                            "Filename": fname,
                            "Image": f"data:image/jpeg;base64,{img['base64']}",
                            "OCR Result": ocr_text
                        })
                else:
                    # エラー等の場合
                      for img in original_group:
                        final_ocr_results.append({
                            "Filename": img['name'],
                            "Image": f"data:image/jpeg;base64,{img['base64']}",
                            "OCR Result": "Error: Batch Response Empty"
                        })

            return final_ocr_results, prompt_logs

        except Exception as e:
            st.error(f"結果ファイルの取得・パースエラー: {e}")
            return [], prompt_logs
    else:
        st.error("出力ファイルIDが見つかりませんでした。")
        return [], prompt_logs


# --- 結果表示ロジック (HTMLテーブル版) ---
def render_result_table(results):
    """結果リストを受け取り、HTMLテーブルを描画する"""
    
    html = '<table class="ocr-table">'
    html += '''
    <thead>
        <tr>
            <th width="50">No</th>
            <th width="150">ファイル名</th>
            <th width="480">画像</th>
            <th>OCR結果</th>
        </tr>
    </thead>
    <tbody>
    '''
    
    for i, row in enumerate(results):
        no = i + 1
        fname = row['Filename']
        ocr_text = row['OCR Result']
        
        if row['Image']:
            img_tag = f'<img src="{row["Image"]}" loading="lazy">'
        else:
            img_tag = '<span style="color:#94A3B8;">画像なし</span>'
            
        safe_text = html_lib.escape(ocr_text)
        
        html += '<tr>'
        html += f'<td align="center" style="font-weight:bold;">{no}</td>'
        html += f'<td style="font-weight:bold;">{fname}</td>'
        html += f'<td>{img_tag}</td>'
        html += f'<td><div class="ocr-text-cell">{safe_text}</div></td>'
        html += '</tr>'
        
    html += '</tbody></table>'
    
    st.markdown(html, unsafe_allow_html=True)


# --- アプリケーションUI本体 ---

# --- サービスアカウント表示（上部） ---
sa_email = "ocr-app@ai-project-427106.iam.gserviceaccount.com" 
try:
    if "google" in st.secrets and "credentials_json" in st.secrets["google"]:
        gcp_dict = json.loads(st.secrets["google"]["credentials_json"])
        if "client_email" in gcp_dict:
            sa_email = gcp_dict["client_email"]
except:
    pass

st.markdown(f"""
<div style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 5px; margin-top: 15px;">
    👇 以下のサービスアカウントを「閲覧者」以上の権限で、対象のGoogleドライブフォルダに共有してください。
</div>
""", unsafe_allow_html=True)
st.code(sa_email, language="text")

# --- URL入力（下部） ---
target_folder = st.text_input("対象フォルダのURL", key="input_folder_run", placeholder="https://drive.google.com/drive/folders/...")

st.write("") # 余白

# --- ボタンクリック時にステートをクリアするコールバック関数 ---
def clear_previous_results():
    """OCR開始ボタンクリック時に即座に結果をクリアする"""
    st.session_state.ocr_results = None
    st.session_state.prompt_logs = None

# STARTボタン
# on_click引数にコールバック関数を指定することで、処理開始前に画面のクリアが反映されます
if st.button("OCRジョブを開始", type="primary", on_click=clear_previous_results):
    if not target_folder:
        st.toast("フォルダURLを入力してください", icon="⚠️")
    else:
        folder_id = extract_folder_id(target_folder)
        
        try:
            creds = get_credentials()
            service = build('drive', 'v3', credentials=creds)
            
            # 1. リスト取得
            with st.spinner("フォルダをスキャン中..."):
                query = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false"
                results = service.files().list(q=query, fields="files(id, name)").execute()
                files = results.get('files', [])

            if not files:
                st.error("画像が見つかりませんでした。")
            else:
                # 2. 並列ダウンロード
                st.toast(f"{len(files)} 枚の画像を検出。ダウンロード開始...", icon="⚡")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                downloaded_images = loop.run_until_complete(download_images_parallel(creds, files))
                
                if not downloaded_images:
                    st.error("ダウンロードに失敗しました。")
                else:
                    st.toast(f"AI解析を開始します (Batch API)...", icon="🧠")
                    
                    # 3. Batch API 実行
                    # run_batch_api_process 内部で upload -> batch create -> polling -> download まで行います
                    ocr_results, prompt_logs = loop.run_until_complete(run_batch_api_process(downloaded_images))
                    
                    if ocr_results:
                        # 4. 結果ソート
                        ocr_results.sort(key=lambda x: x['Filename'])
                        
                        # 5. 結果をセッションステートに保存
                        st.session_state.ocr_results = ocr_results
                        st.session_state.prompt_logs = prompt_logs
                        
                        st.toast("完了しました！", icon="🎉")

        except Exception as e:
            st.error(f"予期せぬエラー: {e}")

# --- 結果表示エリア ---
if st.session_state.ocr_results:
    results = st.session_state.ocr_results
    logs = st.session_state.prompt_logs
    
    # A. プロンプトログ表示 (トグルで開閉)
    st.subheader("実行プロンプト（送信ログ）")
    show_logs = st.toggle("ログ詳細を表示する", value=False)
    
    if show_logs:
        st.info(f"送信されたリクエスト詳細です。1リクエストにつき {IMAGES_PER_GROUP} 枚までの画像を含んでいます。")
        for log in logs:
            group_id = log['group_id']
            files_str = ", ".join(log['files'])
            with st.expander(f"リクエストグループ #{group_id} (対象ファイル: {files_str})"):
                formatted_json = json.dumps(log['messages'], indent=2, ensure_ascii=False)
                st.code(formatted_json, language='json')

    st.divider()

    # B. 結果表示 (HTMLテーブル)
    st.subheader(f"OCR結果一覧（全 {len(results)} 件）")
    render_result_table(results)