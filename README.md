# GAHSO：LEO 衛星網路與 SFC 生命週期

依據本目錄論文的第 III–IV 節、公式 (1)–(13)、第 VII 節與 Table I，
建立可供後續 GAHSO 強化學習模組使用的 Python 環境。
目前預設依使用者要求縮小為 **20 顆衛星、4 個軌道、每軌道 5 顆**。

目前完成：動態衛星圖、SFC 產生器、資源驗證與配置、到期釋放、
斷鏈／延遲超限偵測，以及可原子替換配置的遷移介面；另已完成第 VI-A 節的逐跳 MDP、
公式 (17)/(18) 獎勵，以及可共用的負載、公平性與 C1–C8 診斷介面。
範例採用 shortest-delay + ordered-greedy 策略，**不是 GAHSO 或完整 LASO**；
已實作邊特徵 GAT、Actor／Critic／Target Critic 與逐跳互動；A2C optimizer 訓練與遷移決策尚未實作。

## 執行

核心網路、基準策略與 MDP 使用 Python 標準函式庫；GAT 及其測試需要 PyTorch 與 NumPy。
本機以 Python 3.14、PyTorch 2.14.0+cpu、NumPy 2.5.3 驗證。
執行全部測試前請安裝 requirements-gat.txt；只執行原有模擬器則不需安裝。

在本專案目錄執行：

```powershell
py -3.14 -B -m gahso --config config/default.json
py -3.14 -B -m gahso --config config/default.json --output results/demo_20.json
py -3.14 -B -m gahso --mode mdp-demo --output results/mdp_demo.json
py -3.14 -B -m unittest discover -s tests -v
```

第一個命令列出摘要；第二個保存 20 顆衛星的完整模擬結果；
第三個示範單一請求的逐跳 MDP 與完整 (s,a,r,s_next) 軌跡。
舊的 results/demo.json 是歷史結果，不會因修改預設參數自動更新。
`-B` 避免產生 Python bytecode 快取，並非必要選項。

## 論文參數與補充假設

| 項目 | 預設值 | 依據 |
|---|---|---|
| 星座 | Walker Star，20 顆／4 軌道 | 使用者指定 20 顆；4×5 保留同軌通視。論文主實驗為 100 顆／10 軌道 |
| 軌道高度 | 2,000 km | Table I |
| 衛星 CPU | 400–600 CPU unit | Table I；區間內均勻抽樣為補充假設 |
| 每條星間鏈路頻寬 | 200 MHz | Table I |
| 每個 VNF 計算需求 | 50–200 CPU unit | Table I |
| 每個服務頻寬需求 | 20–80 MHz | Table I |
| 光速 | 299,000 km/s | Table I 的 2.99×10^8 m/s |
| 服務到達率 | 平均 3 個／slot | 第 VII 節的其中一個實驗情境 |
| 頻寬型／計算型數量比 | 0.5 | 第 VII 節；因此頻寬型機率為 1/3 |
| 傾角／相位因子 | 90°／0 | 可調假設，論文未完整指定 |
| 相鄰軌道升交點赤經 | 在 180° 範圍均分 | 本實作的 Walker Star 定義 |
| 地球半徑／重力參數 | 6,371 km／398600.4418 km³/s² | 模擬常數，均可設定 |
| 到達分布 | Poisson process（指數到達間隔） | 補充假設 |
| slot 時間 | 10 秒 | 補充假設 |
| 服務持續時間 | 指數分布，平均 120 秒 | 補充假設 |
| VNF 數量 | 均勻抽樣 3–5 個，含起終點 VNF | 補充假設 |
| 服務延遲上限 | 100 ms | 補充假設 |
| 來源／目的衛星 | 均勻抽樣不同衛星 | 示範假設，未取得論文地面站座標清單 |
| 負載權重 ζ1、ζ2 | 各 1.0 | 可調假設，不是論文已公布校準值 |

所有設定見 `config/default.json`。要測試 25 顆星可設 5 軌道；
200 顆星可設 10 軌道。衛星必須等量分布於各軌道，且至少 3 軌道、每軌道 3 顆。

兩類服務依 Table I 區間的中點分割：
頻寬型 CPU 50–125、頻寬 50–80 MHz；
計算型 CPU 125–200、頻寬 20–50 MHz。此類別分布為明確的補充假設，
不是論文公布的精確抽樣方法。請勿以此環境的接受率當作論文結果重現。

## 網路建模

- 使用圓軌道位置公式與角速度 sqrt(μ/r³)，保存慣性座標位置。
- 同軌道連接前後鄰居；跨軌道連接相鄰軌道。
- 每對相鄰軌道依距離排序後做一對一配對，避免多個衛星選中同一顆衛星
  而超過四個收發器。這是對論文「最近鄰軌衛星」未指定衝突解法的補充。
  此貪婪配對不保證每個節點都取得自身最近的鄰星，也不是全域最小成本匹配。
- 預設允許首尾軌道配對；可用 `close_plane_seam=false` 關閉。
  未加入極區停用、反向軌道接縫操作規則或天線指向限制。
- 預設排除穿越地球與零距離的鏈路。因此收發器數是上限，實際節點度數可低於 4。
- 鏈路為無向邊，兩方向共用 200 MHz 的保留帳本。
  若實驗需要雙向各自容量，應另改為有向資源模型。
- 延遲為距離／光速；對應論文公式 (1)、(6)，不加入排隊、處理或存取鏈路延遲。
- `GroundTerminal.nearest_visible()` 支援經緯度、地球自轉與最低仰角，
  選擇最近可見衛星；無可見衛星時回傳 None。示範流量預設沒有使用真實地面站。
- 星體移動會更新距離與配對；可用 `snapshot(t, disabled_links=[(a,b)])`
  注入指定星間鏈路故障。關閉鏈路可能讓其他可行配對建立。

## SFC 配置與資源生命週期

`SFCRequest` 保存 ID、來源／目的、有序 VNF、CPU／頻寬需求、
延遲上限、到達時間與服務持續時間。到期時間為 arrival + duration。

外部策略回傳 `Allocation(path, vnf_nodes)`；
`vnf_nodes[i]` 是第 i 個 VNF 的衛星。
路徑目前限制為不重複經過節點的簡單路徑；VNF 可共置於同一顆衛星。
第一個 VNF 必須在來源，最後一個在目的；全部 VNF 必須沿路徑依序放置。

配置流程：

1. 檢查請求是否到達／已過期，並排除重複 ID。
2. 檢查路徑、端點、VNF 數量與順序，以及鏈路存在性。
3. 檢查總傳播延遲。
4. 聚合同一顆衛星上的所有 VNF CPU 需求，再檢查 CPU 與鏈路頻寬。
5. 全部驗證通過才一次提交資源；失敗記錄拒絕原因，不扣除任何資源。
6. 服務到期或明確 release 時，恰好釋放一次；重複釋放無副作用。

請求狀態：

```text
新請求 ── 配置成功 ──> active ── 到期 ──> expired
   └───── 配置失敗 ──> rejected
active ── 斷鏈／延遲超限 ──> interrupted
interrupted ── 路徑恢復或替換配置成功 ──> active
active / interrupted ── 主動釋放 ──> released
interrupted ── 到期 ──> expired
```

中斷期間保留舊 CPU 與鏈路的資源保留量，避免舊配置和新配置重複分配；
鏈路消失不會讓帳本遺失。負載指標因此表示「保留資源負載」，不等同實際輸送流量。
`pending_migrations()` 列出中斷請求；
`replace_allocation()` 在扣抵自身既有資源後驗證新配置，
成功才替換，失敗保留舊配置。這是遷移決策的掛接介面，
尚未模擬 live migration 的額外傳輸量、時間或雙份暫存 CPU。

時間由事件驅動。相同時間先到期、再拓樸更新、最後處理新請求。
在固定 topology_step_s、每次到達與到期事件時，都會重新計算網路並檢查存活服務；
事件之間的短暫斷鏈可能未被取樣到，可縮小 topology_step_s。
事件延遲與服務持續時間均以秒表示。

預設 `drain=true`：horizon_s 後停止生成新請求，但持續執行直到工作負載全部到期。
`at_horizon` 是觀察區間終點，`final` 是清空後狀態。
`drain=false` 則在 horizon_s 停止，保留尚未到期服務的資源。

## 基準策略介面

```python
from gahso.simulation import Simulation
from gahso.resources import Allocation

def policy(manager, request):
    # 此處由 GAT/A2C 取得路徑，再完成有序 VNF 放置。
    # 無法配置時回傳 None；否則：
    # return Allocation(tuple(path), tuple(vnf_nodes))
    return None

report = Simulation(policy=policy).run()
```

策略可讀取 `manager.snapshot.satellites`、`manager.snapshot.links`、
`snapshot.neighbors(n)`、`cpu_available(n)` 與 `bandwidth_available(a,b)`。
策略只產生候選配置，提交由模擬器的 ResourceManager 統一完成。
目前為單執行緒環境，原子性指驗證與提交間不會部分修改帳本，沒有提供跨執行緒鎖。

## 輸出與驗證

- admission_ratio：累計接受／累計到達，不代表整段服務持續成功率。
- statuses：active、interrupted、rejected、expired、released。
- cpu_reserved／bandwidth_reserved_mhz：目前保留資源總量。
- load_fairness：依論文公式 (4)、(5)。衛星通訊負載採其相鄰鏈路保留頻寬總和，
  每條鏈路在兩端各計一次；CPU 採已保留量。全空載時定義公平性為 1。
- interruption_events：累計中斷次數。單一服務可多次中斷及恢復。
- events：帶時間、請求 ID 與原因的接受、拒絕、到期、中斷、恢復與重新配置紀錄。

每個模擬事件後重新從服務紀錄推算帳本，檢查帳本一致性與容量限制。
測試另涵蓋圓軌道週期、地球遮蔽、四收發器上限、20/25/100/200 顆星、
隨機種子、VNF 順序、端點、資源不足、原子配置／替換、
重複請求／釋放、同時到期與到達，以及清空後資源歸零。

檔案：
- `gahso/network.py`：星座、位置、鏈路、地面站
- `gahso/services.py`：VNF、SFC 與流量產生器
- `gahso/resources.py`：驗證、資源帳本與生命週期
- `gahso/simulation.py`：事件模擬
- `gahso/baseline.py`：可替換的簡易示範策略
- `gahso/embedding.py`：基準策略與 MDP 共用的有序 VNF 放置
- `gahso/mdp.py`：MDP 状態、動作、轉移、獎勵與終止
- `gahso/mdp_demo.py`：逐跳介面示範
- `tests/test_mdp.py`：公式與 MDP 狀態轉移測試
- `gahso/gat_config.py`：GAT 輸入與模型設定（不依賴 PyTorch）
- `gahso/features.py`：MDP observation → GATInputs 張量
- `gahso/gat.py`：含邊特徵的 GAT 層與多層提取器
- `gahso/gat_demo.py`：GAT 前向／反向傳播示範
- `tests/test_gat.py`：公式、特徵及梯度測試
- `tests/test_environment.py`：測試

## 負載、公平性與可行性 API

```python
loads = manager.node_loads()  # {satellite_id: zeta1 * CPU_used + zeta2 * BW_used}
fairness = manager.fairness() # Jain 指標，含零負載節點
report = manager.check_allocation(request, allocation)  # 只讀，不扣資源
print(report.feasible, report.delay_s, report.violations)
for check in report.checks:
    print(check.code, check.passed, check.reason)
```

公式 (4) 使用已占用資源量；公式 (5) 採用所有衛星的負載。
Jain 指標以最大值縮放後計算，避免大數平方溢位；全空載定義為 1。
公式 (6) 為整條實體路徑延遲。通訊負載仍採相鄰鏈路保留頻寬總和，
同一鏈路在两端各算一次；這是明確的實作定義。

| 約束 | 程式中的意義 |
|---|---|
| C1 / C2 | 第一／最後 VNF 分別在來源／目的節點 |
| C3 | 每個 VNF 恰有一個放置節點 |
| C4 | 總傳播延遲不超過 T_q |
| C5 | 逐一建構相鄰 VNF 的有向子路徑，驗證每節點出流－入流 |
| C6 | 同節點的 VNF CPU 需求加總不得超額 |
| C7 | 每條實體鏈路的頻寬不得超額 |
| C8 | 用單一整數節點選擇表示二元嵌入，拒絕分數節點 ID |

額外檢查到達／到期、重複請求、實體路徑端點、鏈路存在性、簡單路徑及 VNF 順序。
公開 report 可同時包含多個失敗原因；admit 保留第一個失敗原因供生命週期紀錄使用。
projected_loads(request, allocation) 可預覽配置後負載而不修改帳本；
只容許 C4 延遲違反，其餘條件須可行。

## GAHSO MDP 使用方式

```python
from gahso import SFCOrchestrationEnv, SFCRequest, VNF

env = SFCOrchestrationEnv()  # 預設 20 顆衛星
q = SFCRequest(
    id="q0", source=0, destination=2,
    vnfs=(VNF("ingress", 75), VNF("process", 75), VNF("egress", 75)),
    bandwidth_mhz=30, max_delay_s=0.1, arrival_s=0, duration_s=30,
)
state, info = env.reset(request=q)
for next_satellite in (1, 2):
    action = state["node_ids"].index(next_satellite)
    state, reward, terminated, truncated, info = env.step(action)
    print(reward, info)
    if terminated or truncated:
        break
env.advance_time(30)  # 到期後釋放資源

# 下一個請求以 env.start_request(q_next) 開始，保留其他存活服務；
# env.reset() 則清空帳本，開始新的訓練回合。
```

此處提供 Gymnasium 風格的五回傳值，但沒有相依 Gymnasium，也不是 gymnasium.Env 子類別。
將來如需現成 RL 套件，仍需增加 spaces / tensor 轉換的 adapter。
GAT、Actor、Critic、Target Critic 與 rollout 已實作；A2C optimizer 更新尚未實作。

### State：s_t = {G_t, C_t, B_t, l_t, I_t(q)}

- node_ids：固定排序的衛星 ID，所有節點向量依此對齊。
- node_features：[N, 9]，欄位為剩餘 CPU 比率、縮放負載、來源、目的、已走過、
  目前位置、頻寬型、計算型、自訂型。詳細名稱見 node_feature_names。
- edge_index：[2, 2|E|]，端點是 node_ids 的索引；無向 ISL 拆成雙向訊息邊。
- edge_features：[2|E|, 2]，欄位為剩餘頻寬比率、縮放延遲。
  這兩個訊息方向共享同一實體鏈路資源，不會重複扣款。
- cpu_remaining、link_remaining、node_loads：原始資源量與負載。
- request：完整請求（含有序 VNF CPU、頻寬、延遲上限與時間）。
- path、position、step_count、steps_remaining、accumulated_delay_s：編排進度；
  保留剩餘步數才能正確描述有限 K 的決策狀態。
- fairness：目前候選狀態的 Jain 指標。

此為 GAT 前的原始特徵編碼，不等同 Table I 的 128 維隱藏特徵；
目前的 GATFeatureBuilder 會再加入服務需求與路徑順序，GAHSOGAT 輸出 128 維。
load_scale 與 delay_scale_s
只影響特徵，不改變獎勵公式的原始單位。

### Action 與狀態轉移

step 接收整數索引，選擇 node_ids[index] 為下一跳；
對應論文 one-hot 向量中的一個位置。

- action_mask 只遮蔽非鄰居，對應論文定義。
- feasible_action_mask 額外排除回訪及頻寬不足節點，供示範／除錯使用；
  訓練若使用此遮罩，就是比論文更強的動作過濾，應在實驗中註明。
- 非法動作維持位置及路徑，扣一個決策步數，並給 -alpha1。
- 合法動作更新候選路徑、累積延遲及候選頻寬；未到目的地不配置 CPU。
- 到達目的地後以共享的 Algorithm 2 放置器產生配置，通過 C1–C8 才正式提交。
- 來源等於目的時，以該節點的一個本地完成動作放置所有 VNF，頻寬需求為零。

一個 SFC 的逐跳步數 t 是決策時間，**不會讓衛星每走一跳就運行一秒**。
物理時間只在 start_request / advance_time 的服務邊界推進；
同一請求的拓樸與背景帳本固定。因此這是「單一請求編排」的 MDP，
連續請求及動態服務生命週期由外層排程驅動。
決策期間請勿直接從外部修改 env.resources。

### Reward：公式 (17)、(18)

合法非終止步：
`-theta1 * hop_delay_s - theta2 * shortest_hops - theta3 * next_node_load`

非法非終止步：
`-alpha1`

成功終止：
`beta0 + gamma1 * fairness`

到達、VNF 可放置，但延遲超限：
`beta1 - theta4 * total_delay_s + gamma1 * projected_fairness`

其餘失敗：
`-alpha2 - theta5 * shortest_hops - theta6 * unembedded_vnfs + gamma1 * fairness`

- 中間步的 next_node_load 使用「加入本次候選鏈路後」的負載。
- 成功使用正式配置後的公平性；延遲超限使用只讀候選配置的公平性；
  其他失敗使用未接受該請求的正式帳本公平性。失敗一律不殘留資源。
- n_embed 由部分路徑上的有序放置預覽計算，不是直接固定為 VNF 總數。
- 最短跳數在請求開始時以無權 BFS 計算；不連通時使用 N 作有限懲罰距離。
- 到達目的地前不因延遲超限提前終止，使公式 (18) 的超時分支可達。
- 終止步只回傳公式 (18)，不與同一步公式 (17) 相加。
- K 步用盡是論文任務定義的失敗，回傳 terminated=True、truncated=False；
  訓練時此類終止狀態不應 bootstrap。最後一步到達目的地仍可成功。
- 無剩餘可行下一跳時提前失敗；此 dead-end 處理為補充實作選擇。

config/default.json 的 mdp 區塊集中設定 K、特徵縮放與所有 reward 係數。
K 預設 25，位於 Table I 的 20–30；論文未公布的 reward 係數均是可調預設，
不是論文精確超參數。這些實作選擇及小型星座均需在實驗報告中明確區分。

### 檔案註解

核心 Python 模組已補上繁體中文模組／類別／函式說明，以及公式對應、
演算法步驟、單位、候選與正式資源差異等註解。JSON 規格不允許註解，
因此設定欄位的說明集中於本文件與對應的 dataclass。

## GAT 安裝與執行

```powershell
py -3.14 -m pip install -r requirements-gat.txt
py -3.14 -B -m gahso --mode gat-demo
py -3.14 -B -m gahso --mode gat-demo --output results/gat_demo.json
py -3.14 -B -m unittest discover -s tests -v
```

requirements-gat.txt 固定本次驗證的 CPU 版版本；沒有加入 PyTorch Geometric、
torch_scatter 或其他圖套件。基準與 MDP 模式以延後匯入保留輕量執行方式。
示範會列出輸入／輸出維度、注意力正規化誤差、每個參數的梯度，以及是否改動資源。
選擇 --output 時另保存全部節點 embedding 與各層注意力權重。
示範使用隨機初始化與平方均值 loss 驗證反向傳播，**不是已訓練的 GAHSO 策略**。

### GAT 輸入特徵

GATFeatureBuilder 接收完整 MDP observation，不修改原字典或資源帳本。
以預設 max_vnfs=8 為例：

| 張量 | 維度 | 內容 |
|---|---|---|
| x | [N,32] | 基礎節點特徵、路徑順序與廣播的服務需求 |
| edge_index | [2,2|E|] | 第 0 列 sender j；第 1 列 receiver i |
| edge_attr | [2|E|,2] | 剩餘頻寬比率、縮放延遲 |
| request_features | [21] | 完整的固定長度需求編碼 |
| position | [N] | 當前位置 one-hot |
| action_mask | [N]，bool | 論文定義的鄰居動作遮罩 |
| feasible_action_mask | [N]，bool | 額外的可行動作遮罩，未套用至 GAT 注意力 |

32 維節點特徵的次序可直接讀取 graph.node_feature_names：

1. MDP 原有 9 維：剩餘 CPU 比率、負載、來源／目的、已走過、目前位置及服務類型。
2. 絕對剩餘 CPU 1 維：除以 cpu_demand_scale，以區分利用率相同但容量不同的衛星。
3. 路徑順序 1 維：在 path 中的索引 / max_steps。未走過與來源皆為 0，
   但原有 visited 與 is_source 可區分兩者。保留順序可避免僅有 visited 集合
   卻遺失 VNF 順序判斷所需的路徑資訊。
4. 每個節點廣播相同的 21 維請求資訊：
   頻寬需求、延遲上限、VNF 數量、剩餘延遲預算、剩餘步數比率，
   以及 8 個有序 VNF CPU 需求與 8 個 padding mask。

CPU 需求預設除以 200、頻寬除以 200 MHz、請求延遲除以 0.1 秒。
這些是固定尺度，超過範圍不截斷到 1；延遲超限的剩餘預算保留負值。
原有節點／邊特徵的尺度沿用 mdp.load_scale 與 mdp.delay_scale_s；
請求資訊的尺度由 gat_features 設定。
改變 max_vnfs 會改變輸入維度 16 + 2*max_vnfs；
超長 VNF 串列會明確報錯，必須調大設定並重新建立相同維度的模型。
所有逐欄編碼及 padding 方式是對論文未指定細節的實作補充。

### 含邊特徵的注意力

對每條訊息邊 j→i：

```text
h_i = W_x x_i                         # 式 (19)，節點投影
e_ij = W_e edge_attr_ij               # 式 (20)，邊投影
score_ij = LeakyReLU(aᵀ[h_i || h_j || e_ij])
alpha_ij = softmax over j in N(i)∪{i} # 式 (21)
output_i = sum_j alpha_ij * h_j       # 式 (22)
```

- 原文寫 sigmoid，但公式 (21) 實際為 softmax；依公式實作。
- 訊息依接收節點 i 分組，避免錯在所有邊一起正規化或沿 sender 正規化。
- 每顆衛星加入恰好一個自迴圈，使孤立節點仍有自身表示。
  自迴圈邊特徵設為零，只用於模型內部；不改動 MDP 拓樸與頻寬。
  如輸入已含自迴圈，會移除後統一加入，避免重複計算。
- 投影後邊特徵參與打分，聚合的 message 則為節點特徵 h_j，
  對應公式 (22)，不額外把邊向量直接加進 message。
- 每層節點投影亦作為該層 message 投影；每層分別對原始邊特徵做投影。
- 預設 2 層、單頭、hidden_dim=128；兩層之間使用 ELU，最後一層不加 activation。
  num_layers=2、斜率 0.2、自迴圈零值與層間 ELU 是明確的實作選擇。
  可選多頭；每頭輸出取平均，最終維度仍是 [N,hidden_dim]。
- dropout 預設 0；若啟用，僅作用在層間節點表示，回傳的注意力仍加總為 1。
- PyTorch 以稀疏邊列表搭配 scatter_reduce / scatter_add / index_add 實作，
  不建立 N×N 的 dense 注意力矩陣。相關 API 參閱
  [scatter_reduce_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.scatter_reduce_.html)
  與 [index_add_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.index_add_.html)。

### 在程式中使用

```python
import torch
from gahso.features import GATFeatureBuilder
from gahso.gat import GAHSOGAT
from gahso import SFCOrchestrationEnv, SFCRequest, VNF

env = SFCOrchestrationEnv()
q = SFCRequest("q0", 0, 2, (VNF("in", 50), VNF("work", 150), VNF("out", 75)),
               30, 0.1, 0, 30, "compute")
state, _ = env.reset(request=q)

builder = GATFeatureBuilder()
graph = builder.build(state)
model = GAHSOGAT()
embedding, attention = model(graph, return_attention=True)
assert embedding.shape == (20, 128)

# Actor/Critic 已封裝獨立 GAT，另接 graph.position 與動作遮罩。
# 以下只示範可微分性；A2C 的 policy/value loss 仍待下一階段加入。
loss = embedding.square().mean()
loss.backward()

# 推論時停用 dropout 並避免保存梯度。
model.eval()
with torch.no_grad():
    embedding = model(graph)
```

若要改用 float64，可同時執行 model.double() 與 graph.to(dtype=torch.float64)。
張量與模型必須位於相同 device 與 dtype。
本次實際驗證為 CPU；GPU 需另安裝相符 PyTorch 並以 model.to(device)、
graph.to(device) 搬移。尚未提供混合精度或多圖 batching adapter。
相同模型可處理不同 N 的單張圖，無需因衛星數量改變而重建參數。

回傳 attention[k].edge_index 包含模型自迴圈，
attention[k].alpha 形狀為 [訊息邊數+N, heads]。
除明確移除的輸入自迴圈外，實體訊息邊順序保持不變。

### 本階段驗證

測試涵蓋：有序需求／padding／特徵與索引對齊、一步後候選資源變化、
手算公式 (19)–(22)、邊特徵改變注意力及輸出、每接收端／每頭權重加總、
孤立節點與零邊圖、節點排列等變性、非鄰居不傳訊息、穩定 softmax、
數值梯度檢查、每層節點與邊投影的非零有限梯度、可重現推論與參數載入、
20/25/100/200 顆衛星輸出維度，以及原有生命週期與 MDP 回歸測試。


## Actor、Critic 與逐跳互動

執行未訓練策略的真實逐跳取樣：

```powershell
py -3.14 -B -m gahso --mode actor-critic-demo
py -3.14 -B -m gahso --mode actor-critic-demo --output results/actor_critic_demo.json
py -3.14 -B -m unittest discover -s tests -v
```

Actor 與 Critic 各自使用獨立的 GAT，將 [N,128] 圖表示展平，
串接 N 維目前位置 one-hot，再通過三層 Linear MLP。
Actor 輸出 N 個 logits（預設 20）；Critic 輸出純量 V(s)。
MLP 隱藏寬度 128／64、ReLU 與 Target 同步間隔 10 是可設定的工程選擇；
discount=0.9999 採論文 Table I。設定在 config/default.json 的 agent。
展平架構需要固定 N；更改衛星數後必須重建三個網路。

Actor 只遮罩非鄰居，使用 [Categorical](https://docs.pytorch.org/docs/2.14/distributions.html#categorical)
建立分布，再依專用 RNG 取樣下一跳；deterministic=True 則取最大機率。
回訪或頻寬不足的鄰居仍可選，交由 MDP 的非法動作獎勵處理。
完全無鄰居時記錄 action=None 的失敗終止，不建立全遮罩 softmax，
也不捏造 policy log_prob。

Target Critic 是線上 Critic（含 GAT）的完整獨立副本，始終 eval、凍結參數，
並以 [no_grad](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html) 評估。
agent.sync_target() 可立即同步；未來每次真正執行 Critic optimizer.step() 後，
呼叫 agent.notify_critic_updated()，每 target_sync_interval 次更新執行 hard copy。
互動本身不更新參數或增加此計數；計數包含在 state_dict。

```python
from gahso import SFCOrchestrationEnv, SFCRequest, VNF
from gahso.actor_critic import ActorCriticAgent
from gahso.rollout import RolloutCollector

env = SFCOrchestrationEnv()
agent = ActorCriticAgent(num_nodes=len(env.node_ids))
collector = RolloutCollector(env, agent, seed=11)
q = SFCRequest("q0", 0, 2, (VNF("in", 50), VNF("work", 150), VNF("out", 75)),
               30, 0.1, 0, 30, "compute")
rollout = collector.collect(q)
report = rollout.to_dict()
env.advance_time(q.expires_s)  # 到期釋放已接受請求的 CPU／頻寬
```

collector 預設保留其他請求的資源；reset_network=True 才清空環境。
每一步保存 s、a、r、s_next、log_prob、entropy、V(s)、V_target(s_next)、
TD target 與 advantage。預設關閉 autograd；track_grad=True 保留 Actor／Critic
計算圖供請求結束後更新，仍不自動執行 optimizer。
收集期間關閉 dropout，結束後恢復原來的訓練模式。

TD target = r + discount × V_target(s_next)，終止時僅為 r。
成功、失敗與 K 步數耗盡均不 bootstrap。
論文式 (23) 的 TD error 為 V−target；此處同時提供常見 policy advantage=target−V，
兩者符號相反。target 和 advantage 均 detach，避免梯度流入 Target 或混入另一個網路。

獎勵沿用式 (17)、(18)：中間步包含延遲、剩餘跳數、負載；
非法動作受罰；終止分成功、延遲違反與其他失敗並加入公平性。
info.reward_components 列出實際回傳獎勵的分項，分項加總等於 reward。
終止獎勵取代當步即時獎勵；若有 instant_reward_components，
它僅供診斷，不再加入總獎勵。只有可行且成功的配置才正式保留資源。

此示範使用未訓練的 Actor，因此可能失敗；不代表 GAHSO 訓練後的效能。
目前沒有 optimizer、完整 A2C 訓練迴圈或論文效能重現。
