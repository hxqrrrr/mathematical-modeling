import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib
import tkinter as tk
from tkinter import messagebox
import os

# 设置模型文件路径
model_path = 'C:\\Users\\hxq11\\Desktop\\mathematical-modeling\\模型大全\\预测\\BP神经网络\\my_bp_model.pth'

# 检查文件是否存在并加载模型
if os.path.exists(model_path):
    print(f"模型文件存在于: {model_path}")
    try:
        # 加载模型
        model = torch.load(model_path)
        print("模型加载成功")
        # 将模型设置为评估模式
        model.eval()
    except Exception as e:
        print(f"加载模型时出错: {e}")
else:
    print(f"错误：模型文件不存在于 {model_path}")

# 打印 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 加载 scaler
X_scaler = joblib.load('X_scaler.joblib')
y_scaler = joblib.load('y_scaler.joblib')

def predict_vo2_max():
    try:
        # 获取输入值
        age = float(age_entry.get())
        gender = 1 if gender_var.get() == "男" else 0
        weight = float(weight_entry.get())
        height = float(height_entry.get())
        resting_hr = float(resting_hr_entry.get())
        systolic_bp = float(systolic_bp_entry.get())
        diastolic_bp = float(diastolic_bp_entry.get())
        max_hr = float(max_hr_entry.get())

        # 创建输入数组
        input_data = np.array([[age, gender, weight, height, resting_hr, systolic_bp, diastolic_bp, max_hr]])

        # 数据预处理
        input_scaled = X_scaler.transform(input_data)

        # 转换为 PyTorch tensor
        input_tensor = torch.FloatTensor(input_scaled)

        # 预测
        with torch.no_grad():
            prediction_scaled = model(input_tensor)

        # 转换回 numpy 数组
        prediction_scaled_np = prediction_scaled.numpy()

        # 反向转换预测结果
        prediction = y_scaler.inverse_transform(prediction_scaled_np)

        # 显示结果
        result = f"预测的最大吸氧量为: {prediction[0][0]:.2f} ml/kg/min"
        messagebox.showinfo("预测结果", result)

    except ValueError:
        messagebox.showerror("错误", "请确保所有输入都是有效的数字！")
    except Exception as e:
        messagebox.showerror("错误", f"发生错误：{str(e)}")

# 创建主窗口
root = tk.Tk()
root.title("最大吸氧量预测器")

# 创建输入字段
tk.Label(root, text="年龄:").grid(row=0, column=0)
age_entry = tk.Entry(root)
age_entry.grid(row=0, column=1)

tk.Label(root, text="性别:").grid(row=1, column=0)
gender_var = tk.StringVar(value="男")
tk.Radiobutton(root, text="男", variable=gender_var, value="男").grid(row=1, column=1)
tk.Radiobutton(root, text="女", variable=gender_var, value="女").grid(row=1, column=2)

tk.Label(root, text="体重 (kg):").grid(row=2, column=0)
weight_entry = tk.Entry(root)
weight_entry.grid(row=2, column=1)

tk.Label(root, text="身高 (cm):").grid(row=3, column=0)
height_entry = tk.Entry(root)
height_entry.grid(row=3, column=1)

tk.Label(root, text="静息心率:").grid(row=4, column=0)
resting_hr_entry = tk.Entry(root)
resting_hr_entry.grid(row=4, column=1)

tk.Label(root, text="收缩压:").grid(row=5, column=0)
systolic_bp_entry = tk.Entry(root)
systolic_bp_entry.grid(row=5, column=1)

tk.Label(root, text="舒张压:").grid(row=6, column=0)
diastolic_bp_entry = tk.Entry(root)
diastolic_bp_entry.grid(row=6, column=1)

tk.Label(root, text="最大心率:").grid(row=7, column=0)
max_hr_entry = tk.Entry(root)
max_hr_entry.grid(row=7, column=1)

# 创建预测按钮
predict_button = tk.Button(root, text="预测", command=predict_vo2_max)
predict_button.grid(row=8, column=0, columnspan=2)

# 运行主循环
root.mainloop()
pip uninstall torchY