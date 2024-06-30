from ultralytics import YOLO

# 加載 YOLOv8 模型
model = YOLO('yolov8n.pt')  # 替換為您的模型文件路徑

# 訓練模型
model.train(data='data.yaml', epochs=50, batch=8)  # 將 batch_size 改為 batch

# 評估模型
results = model.val()

# 打印評估結果
print(results)

# 使用模型進行推理
results = model('test/')  # 替換為您的測試圖像資料夾路徑
results.save()

