#!/bin/bash

# --- 腳本設定 ---
CHECKPOINT_DIR="checkpoint" # 你的 checkpoint 資料夾路徑
# 確保腳本在執行前有權限 (chmod +x cleanup_checkpoints.sh)

# --- 函式定義：顯示使用說明 ---
usage() {
    echo "使用方式: $0 [-scene_name]"
    echo "範例 1 (清除所有檔案): $0"
    echo "範例 2 (清除特定場景的檔案): $0 -sceneA"
    echo "範例 3 (清除另一個場景的檔案): $0 -scene_01"
    echo ""
    echo "說明: 此腳本用於清除 $CHECKPOINT_DIR/ 資料夾內的模型檢查點檔案。"
    echo "      如果不帶參數，則清空 $CHECKPOINT_DIR/ 內所有檔案。"
    echo "      如果帶參數，則清除包含該參數名稱的 .ckpt 檔案。"
    exit 1
}

# --- 檢查目標資料夾是否存在 ---
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "錯誤: 檢查點資料夾 '$CHECKPOINT_DIR' 不存在。"
    exit 1
fi

# --- 處理參數 ---
# 檢查是否有參數傳入
if [ -n "$1" ]; then
    # 移除開頭的 '-' 符號，作為場景名稱
    SCENE_NAME="${1#-}"
    
    # 檢查場景名稱是否為空
    if [ -z "$SCENE_NAME" ]; then
        usage
    fi

    # 清除特定場景的檔案
    echo "將清除 $CHECKPOINT_DIR/ 中包含 '$SCENE_NAME' 的 .ckpt 檔案..."
    
    # 使用 find 搭配 rm 確保安全且處理空格 (如果有需要)
    find "$CHECKPOINT_DIR" -type f -name "*${SCENE_NAME}*.ckpt" -delete

    # 檢查是否有檔案被刪除
    if [ $? -eq 0 ]; then
        echo "✅ 清除完成！"
    else
        echo "❌ 發生錯誤，或沒有找到符合 '*${SCENE_NAME}*.ckpt' 的檔案。"
    fi
else
    # 沒有參數，清空整個資料夾
    echo "⚠️ 警告: 由於沒有指定參數，將清空整個 '$CHECKPOINT_DIR' 資料夾的內容！"
    
    # 詢問使用者確認
    read -r -p "你確定要清除 $CHECKPOINT_DIR/ 裡的所有內容嗎? (y/N) " response
    
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        # 清空資料夾內容 (保留資料夾本身)
        rm -rf "$CHECKPOINT_DIR"/*
        echo "✅ 清空完成！"
    else
        echo "操作已取消。"
    fi
fi

exit 0