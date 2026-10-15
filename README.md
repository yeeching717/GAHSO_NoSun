# GAHSO：25 顆 LEO 衛星與延遲受限 SFC 編排

正式流程包含邊特徵 GAT、Actor、Critic、Target Critic、逐跳路徑決策、
到達目的地後的有序 VNF 放置、每個請求一次更新，以及部署時的動態修復。
工程假設及調整列於下方，不宣稱完全重現論文實驗。

## 安裝與執行

在專案根目錄，以 Python 3.14 執行：

```powershell
py -3.14 -m pip install -r requirements.txt
py -3.14 -B -m unittest discover -s tests -q
py -3.14 -B -m gahso.evaluate --epochs 5 --requests-per-epoch 20 --output-dir results/validation_25 --plots
py -3.14 -B -m gahso.evaluate --epochs 50 --requests-per-epoch 100 --output-dir results/final_25 --plots
py -3.14 -B -m gahso.deploy --checkpoint results/final_25/policy.pt --output results/deployment.json
```

`py -3.14 -B -m gahso` 與 `gahso.evaluate` 共用正式入口。
唯一正式設定為 `config/default.json`。`--config` 可指定另一份實驗設定。
使用 `--training-mode persistent` 可跨輪保留網路資源及接續到達時間；
預設 `epoch_reset` 每輪重設，輪內請求保持背景負載，到期釋放 CPU/頻寬。

延遲遮罩對照實驗：

```powershell
py -3.14 -B -m gahso.evaluate --epochs 50 --requests-per-epoch 100 --disable-delay-mask --output-dir results/ablation_25 --plots
```

開／關遮罩使用相同初始化、訓練請求種子、測試流量與故障序列，皆從頭訓練。
特徵、reward 及其他設定相同。策略不同會使後续接受集合及資源狀態分歧。
保留各自最終 checkpoint；不根據測試集挑選模型，不混入 shortest-delay fallback。

## 環境及服務

| 項目 | 設定 |
|---|---|
| 星座 | Walker Star；25 顆、5 軌道、每軌道 5 顆 |
| 高度／傾角 | 2,000 km／85 度 |
| 軌道升交點分布 | 在 180 度內均分；相位因子 0 |
| 衛星 CPU | 每星固定從 Uniform(400, 600) 抽樣 |
| 每條鏈路頻寬 | 200 MHz；雙向共用容量 |
| SFC 類型 | 唯一共同分布，無 bandwidth/compute 分支或類型 one-hot |
| VNF 數量 | 均勻抽取整數 3–5，包含起點與終點 VNF |
| VNF CPU | 每個 VNF 獨立 Uniform(50, 200) |
| SFC 頻寬 | Uniform(20, 80) MHz |
| 到達 | Poisson；每 10 秒平均 3 個請求 |
| 生命週期 | 指數分布，平均 120 秒 |
| 延遲上限 | 0.1 秒；本模型計算傳播延遲 |
| 決策步數上限 | 25；與最多 5 個 VNF 不同 |

端點從不同衛星均勻抽樣。類型統一不表示每個請求需求相同。
SFCRequest 及產生器拒絕長度不在 3–5 的請求；GAT 最多編碼 5 個，不靜默截斷。
模型輸出維度依衛星數建立；checkpoint v2 記錄節點數及特徵版本。
舊特徵版本必須重訓；部署會明確拒絕將 20 顆 checkpoint 用於 25 顆環境。

## 四鄰居與通視

同軌道前後各一顆，slot 首尾環接。軌道也首尾環接：每顆在左右相鄰軌道各配對一顆。
每對相鄰軌道以 Hungarian 演算法求一對一最小總歐氏距離配對。
各星獨立選最近鄰可能選到相同對象，無法普遍同時保證雙向一致及每星四鄰居；
一對一最小總距離不等於每顆各自的絕對最近鄰。

名義圖每星恰好四個不同鄰居，共 50 條無向邊。實際圖從名義圖移除地球遮蔽、
距離小於 1e-6 km 的重合點鏈路及故障邊，不另補邊湊足度數。
因此實際度數可以小於四；對特定請求的可行動作還要通過資源與延遲條件。
`report.json/topology_audit` 在一個軌道週期的 17 個快照驗證名義度數、
實際度數、雙向一致性、距離與通視，並統計獨立最近鄰衝突及不可用原因。
`close_plane_seam=false` 可關閉軌道首尾環接，此時邊界軌道名義度數為三。

## 延遲可行性與失敗歸因

`routing.py` 以 Dijkstra 計算剩餘傳播延遲下界。每個候選動作必須滿足：

```
已累積延遲 + 本跳延遲 + 下一節點至目的地的剩餘延遲下界 <= 請求延遲上限
```

剩餘圖排除所有已訪節點（包含目前節點），以及頻寬不足的邊。
找不到剩餘路徑即不可行。此條件不代表 CPU 足夠或 VNF 一定可放置；
到達目的地後仍執行有序放置及完整 C1–C8 檢查，成功後原子提交資源。
`action_mask` 只表示鄰居；Actor、rollout、部署都使用 `feasible_action_mask`。
GAT 注意力仍讀完整可用實體圖；遮罩不會以最短路徑直接替 Actor 選動作。

每請求到達時分別計算無頻寬條件及有頻寬條件的最短延遲：

- `initial_no_physical_route`：實體圖無路。
- `initial_delay_infeasible`：實體最短路徑已超限。
- `initial_bandwidth_infeasible`：實體可達但頻寬限制後無路。
- `initial_bandwidth_delay_infeasible`：頻寬限制後最短路徑超限。
- `initial_endpoint_cpu_infeasible`：路徑可行，但起終點強制 VNF 的 CPU 不足。
- `policy_delay_failure`：初始路徑可行，所選路徑到達後超限。
- `vnf_placement_failure`：路徑與端點 CPU 初步可行，但完整 VNF 放置失敗。
- `policy_route_or_placement_failure`：其他未完成路徑／配置失敗。

這是依序歸因，不是全域 SFC 可行性的數學證明。到達診斷亦保留獨立的端點 CPU 布林值。
終止 info 中的 `blocked_actions` 保留每個鄰居所有適用的拒絕原因。
無實體鄰居會記錄 `no_physical_neighbors`；無可行動作不建立全遮罩 softmax，
以 action=None、terminated=True 結束，無 bootstrap；先前動作仍參與本請求更新，
無動作狀態只學習 Critic，不捏造 log_prob。候選路徑不會占用正式帳本。

## 特徵、reward 與訓練

每節點 25 維：8 個 MDP 特徵、CPU 絕對剩餘量/200、路徑順序/25，
再加 15 維請求特徵。8 個 MDP 特徵為 CPU 剩餘比例、負載/1000、
起點、終點、已訪、目前位置、剩餘延遲下界/請求上限（截至 5）及可達指標。
不可達節點的下界特徵填 0，以可達指標 0 區分；不輸入 infinity。
請求特徵包含頻寬/200、延遲上限/0.1、VNF 數/5、剩餘延遲/0.1、
剩餘步數比例、5 個 CPU/200 與 5 個 padding 指標。
每條有向邊含剩餘頻寬比例及延遲/0.1；无向實體邊轉為兩條訊息邊。

本次正式 reward 延續前一修正版，沒有為提高數字另改尺度。
成功終止：`5 - min(總延遲/上限, 5) + 0.25*Jain`。
失敗基礎懲罰 -5，再扣正規化延遲或未完成成本；逐跳成本使用延遲比例、
hop 距離/節點數/最大步數及容量加權負載比例/最大步數。
`RewardConfig` 的 legacy 模式僅保留為既有手算公式測試參照，正式設定明確指定 normalized。

reward 在訓練 target 中再除以固定 5，兩個 optimizer 均為 Adam、lr=1e-4，
entropy=0.01、gradient clip=5。Critic 使用平均平方 TD 誤差；Actor 使用
detach 後的 advantage 標準化與平均 loss。`update_requests_per_batch=8` 時會在八個完整請求的所有 policy step 上共同標準化；設為 1 可重現逐請求更新。單步/常數 advantage 保留方向，
不因扣平均變成全零。每個請求只更新一次，Target 每 10 次 Critic 更新硬同步。
terminated 不 bootstrap；外部 truncated 保留 bootstrap。
Actor loss 可以為負；單種子平穩窗口不是收斂證明。

## 動態修復與指標

新請求拒絕不等於既有服務中斷。既有服務斷鏈後由遷移規劃器嘗試原子替換配置，
失敗保持中斷狀態及原保留資源，直到恢復、後續修復或到期釋放。
修復是有搜索上限的 sequential search，非全域最優；未模擬狀態傳輸量、
傳輸耗時及遷移期間雙份 CPU，因此持續率只適用此抽象環境。

| 指標 | 分母／意義 |
|---|---|
| acceptance_rate | 接受 SFC 數／到達 SFC 數 |
| affected_services | 生命週期內至少一次修復前中斷的不同服務數 |
| repair_success_rate | 成功修復次數／嘗試次數；同服務可能多次嘗試 |
| service_time_availability | ACTIVE 服務秒／已接受且尚未到期服務秒 |
| uninterrupted_after_repair_rate | 所有檢查事件修復後皆未中斷的接受服務數／接受數 |
| vnf_migrations | 成功配置替換中，實際換衛星的 VNF 數 |
| reroutes | 成功替換中，路徑發生變化的次數 |
| vnf_migrations_per_accepted_sfc | VNF 遷移數／接受 SFC 數 |
| vnf_migrations_per_affected_sfc | VNF 遷移數／受影響 SFC 數 |

零分母回傳 null。策略使用相同流量及故障序列，但接受集合不同，持續率分母亦不同。
遷移數增加不單獨代表品質提高。Jain 取每次到達後全網資源負載的公平性平均。
部署每 100 SFC 時間包含策略、特徵、遮罩及放置／提交，不含訓練、載入及修復。

## 產物及清理

`results/final_25/` 保存正式模型、完整報告、training.csv、training_requests.csv、
comparison.csv 與 PNG/SVG 曲線。`results/ablation_25/` 保存等量遮罩關閉對照。
比較摘要為 `results/comparison_summary.md`；舊 20 顆數字保留於 `results/history_20_summary.md`。
詳細清理清單在 `results/cleanup_manifest.json`，驗證摘要在 `results/verification.json`。
不再提供 demo CLI，測試所需整合工具位於 tests/helpers.py。
舊模型、短訓練產物與快取僅在完成比較摘要後刪除；論文 PDF、baseline、正式模組及回歸測試保留。

與論文的工程差異包括 25 顆規模、單一均勻服務分布、軌道配對衝突解法、
更強的延遲可行性遮罩與下界特徵、正規化 reward、Adam/advantage 穩定化，
以及有上限的原子遷移抽象。這些設定及各實驗完整參數都記錄在 report.json。

## 85° 訓練診斷與端點對照

正式預設為 `endpoint_placement_mode="fixed"`，第一與最後 VNF 分別固定在 source 與 destination。`neighbor_allowed` 只用於工程對照：首尾 VNF 也可放在所選路徑上、與對應端點直接相鄰的節點；它仍需通過 CPU、頻寬、延遲、順序與完整資源檢查，不能視為論文原始限制。

訓練與部署報告包含端點可用 CPU、首尾需求、逐 VNF 可行候選數與放置失敗原因。overall success 以全部請求為分母；initially feasible 需同時通過初始路徑、頻寬／延遲及端點 CPU；conditional success 只以 initially feasible 請求為分母；embedding success 只以 Actor 已到達目的地的請求為分母。分母為零時輸出 `null`。

固定 validation workload 使用獨立固定種子，每 5 epoch 以 greedy Actor 評估凍結拓樸、軌道變動及隨機斷鏈。Actor loss 可為負且不要求單調下降；收斂診斷同時檢查 conditional success、TD error、Critic loss 與 entropy。

```powershell
py -3.14 -B -m gahso.evaluate --epochs 50 --requests-per-epoch 100 --endpoint-placement-mode fixed --update-requests-per-batch 1 --output-dir results/85_formal_A_fixed_batch1 --plots
py -3.14 -B -m gahso.evaluate --epochs 50 --requests-per-epoch 100 --endpoint-placement-mode fixed --update-requests-per-batch 8 --output-dir results/85_formal_B_fixed_batch8 --plots
py -3.14 -B -m gahso.evaluate --epochs 50 --requests-per-epoch 100 --endpoint-placement-mode neighbor_allowed --update-requests-per-batch 8 --output-dir results/85_formal_C_neighbor_batch8 --plots
```
