import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns
from matplotlib import font_manager, rc # 폰트 세팅을 위한 모듈 추가
# font_path = "/home/sml09181/proj-dvd/Ear-VoM/MALGUN.TTF" # 사용할 폰트명 경로 삽입
# font = font_manager.FontProperties(fname = font_path).get_name()
# rc('font', family = font)


# 모든 카테고리, 보이스피싱, 대한민국, 웹 검색
root = "/home/sml09181/proj-dvd/Ear-VoM/prevent/x-vector-pytorch/data"
df1 = pd.read_csv(os.path.join(root, "acc_train.csv"))
df2 = pd.read_csv(os.path.join(root, "acc_valid.csv"))
df3 = pd.read_csv(os.path.join(root, "loss_train.csv"))
df4 = pd.read_csv(os.path.join(root, "loss_valid.csv"))

fig = plt.figure()
fig.set_facecolor('white')
sns.color_palette("crest", as_cmap=True)
sns.lineplot(x=df1['Step'], y=df1['Value'], label="Training")
sns.lineplot(x=df2['Step'], y=df2['Value'], label="Validation")
#plt.yscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.title("Accuracy")
plt.savefig(os.path.join(root, "Accuracy.png"))
plt.close()


sns.lineplot(x=df3['Step'], y=df3['Value'], label="Training")
sns.lineplot(x=df4['Step'], y=df4['Value'], label="Validation")
plt.xlabel('Epoch')
#plt.yscale('log')
plt.ylabel('Loss')

#plt.title("Loss")
#plt.set_title("Training Loss")
plt.show()
plt.savefig(os.path.join(root, "Loss.png"))
plt.close()



exit(-1)

for _, label in enumerate(labels):
    x = range(10)
    y = [random.randint(1,10) for _ in range(10)]
    sns.lineplot(x=x, y=y, label=label) # label 범례 라벨
    plt.legend()

    



# print(df['Count'].sum())
# print(df['Count'].sum()/len(df)/31)
# Sample DataFrame (you would already have this)

# Create a new column for Year.Month
df['Year.Month'] = df['Year'].astype(str) + '.' + df['Month'].astype(str)

# Create a seaborn line plot
plt.figure(figsize=(24, 6))
sns.lineplot(data=df, x='Year.Month', y='Count', marker='o')
plt.title('Voice Phishing Monthly Status(2018.12~2023.09)')
plt.xlabel('Year.Month')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-ticks for better readability
plt.grid()
plt.show()
plt.savefig(os.path.join(root, "police.png"))

exit(-1)

df1 = pd.read_csv(os.path.join(root, "kor.csv")).iloc[:-1, :]
df2 = pd.read_csv(os.path.join(root, "eng.csv")).iloc[:-1, :]

# print(df1.value)
# kor = []
# eng = []
# date = []
# for i in range(len(df1)):pass
print(df1.index.shape, df1.values.shape)

plt.plot(df1.index, np.squeeze(df1.values), color="#8D99AE", marker='o')
plt.plot(df2.index, np.squeeze(df2.values), color="#EF233C", marker='x')
plt.title('Trends in Google search volume')
plt.xlabel('Date')
plt.ylabel('Search Volume')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig(os.path.join(root, "google.png"))

# fig, ax = plt.subplots(figsize=(6, 5))

# # # Plot the baseline
# # ax.plot(
# #     [x[0], max(x)],
# #     [baseline, baseline],
# #     label="Baseline",
# #     color="lightgray",
# #     linestyle="--",
# #     linewidth=1,
# # )

# # -------------------BEGIN-CHANGES------------------------
# # Plot the baseline text
# ax.text(
#     x[-1] * 1.01,
#     baseline,
#     "Baseline",
#     color="lightgray",
#     fontweight="bold",
#     horizontalalignment="left",
#     verticalalignment="center",
# )
# # --------------------END CHANGES------------------------

# # Define a nice color palette:
# colors = ["#2B2F42", "#8D99AE", "#EF233C"]

# # Plot each of the main lines
# for i, label in enumerate(labels):
#     # Line
#     ax.plot(x, y[i], label=label, color=colors[i], linewidth=2)

#     # -------------------BEGIN-CHANGES------------------------
#     # Text
#     ax.text(
#         x[-1] * 1.01,
#         y[i][-1],
#         label,
#         color=colors[i],
#         fontweight="bold",
#         horizontalalignment="left",
#         verticalalignment="center",
#     )
#     # --------------------END CHANGES------------------------

# # Hide the all but the bottom spines (axis lines)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
# ax.spines["top"].set_visible(False)

# # Only show ticks on the left and bottom spines
# ax.yaxis.set_ticks_position("left")
# ax.xaxis.set_ticks_position("bottom")
# ax.spines["bottom"].set_bounds(min(x), max(x))
# # -------------------BEGIN-CHANGES------------------------
# ax.set_xticks(np.arange(min(x), max(x) + 1))
# # --------------------END CHANGES------------------------

# ax.set_xlabel("Size (m^2)")
# ax.set_ylabel("Efficiency (%)")
# # plt.legend()  # REMOVE THE ORIGINAL LEGEND
# plt.show()