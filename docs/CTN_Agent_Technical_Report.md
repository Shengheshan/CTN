# CTN Agent 技术报告：从环境快照到在线策略学习

> **项目**：Capture the Narrative（CTN）虚构社交平台 Agent  
> **代码位置**：CTN-bot 项目根目录  
> **归档版本**：2026-11-03；代码、SQLite 行动记录和官方活动导出的最终计分复核  
> **读者**：需要理解设计动机、具体算法、工程实现及实测边界的开发者与面试官

## 摘要

CTN Agent 是一个在虚构 Kingstonia 社交环境中运行的自动代理。它周期性读取帖子、新闻和账户通知，保存原始与标准化快照，从历史快照重建 WorldState，编码为分层数值特征，生成候选行动，再由启发式策略或上下文 LinUCB 加 Softmax 选择行动。文本行动调用平台提供的大模型生成并审核内容；执行器负责平台写入和幂等控制；反馈系统按固定时间窗口采集行动自身的公开互动指标，生成版本化奖励并记录参数更新。

本项目的核心工程选择是把**环境事实、数值决策、文本生成、平台执行、行动反馈**分开。这样每轮行动可以追溯到快照、候选集、评分、选择概率、生成内容、远端结果和奖励。最终活动导出的 16 条计分记录合计 **1311 分**，报告了 123 次计分浏览、14 次点赞、7 次回复、1 次关注，以及 2 次朝本方和 5 次朝中立方向的观点变化贡献。它们是平台按批次汇总的观察结果，不能归因到某个 Prompt、候选动作或策略更新。数据库确有在线更新记录，但持久化模型样本计数与更新记录不完全一致，第 10 节给出复核口径和限制。

**关键词**：状态建模；动态主题；分层编码；上下文赌博机；LinUCB；Softmax；在线反馈；决策追踪。

---

## 1. 项目背景与问题定义

CTN 是一个受控的虚构社交平台环境。Agent 可以阅读 Legit Social 帖子和新闻站点文章，并通过账户执行 CREATE_POST、REPLY、LIKE、REPOST、FOLLOW、MENTION 或 WAIT。项目中的人物、H2D2 公共议题和公投均属于模拟环境。

目标是构建一个可连续运行、可解释、可复现的代理。单次大模型调用直接决定并发布内容，会把以下问题混在一起：看到什么、选谁互动、何时发、写什么、是否写入成功、后续反馈如何。因此系统把它们拆成明确的状态和接口：

\[
\text{观察} \rightarrow \text{WorldState} \rightarrow \text{Encoder}
\rightarrow \text{候选与策略} \rightarrow \text{ActionPlan}
\rightarrow \text{生成与执行} \rightarrow \text{反馈与学习}.
\]

这里的“学习”特指用当前版本的**行动自身反馈代理值**调整策略参数；它没有直接观测个人投票变化、NPC 身份收益或官方比赛得分。

## 2. 主要思路如何形成

| 阶段 | 当时问题 | 采用的设计 | 当前实现状态 |
|---|---|---|---|
| 环境层 | API 返回结构不同，重启后状态难复现 | RawSnapshot、标准化 Post/Article/Event、SQLite 快照、按快照重建状态 | 已实现 |
| 语义层 | 预设主题表难适应新议题 | 两阶段动态 Topic Registry：批内提取合并、跨批次匹配复用 | 工程流程已实现；语义质量仍不稳定 |
| 表示层 | 对象数量变化，单个固定大向量不合适 | Global/Post/Event/Topic 分层固定维度编码，显式缺失标志 | 已实现 |
| 决策 v0.2 | 无可信奖励时不能直接训练 | 可解释的 CandidateGenerator、PolicyScorer、DecisionEngine 和 WAIT | 已实现 |
| 闭环 v0.3 | 仅凭“回复了”无法评价内容或时机 | ActionPlan、执行日志、定时反馈、内容评估、版本化 Reward、完整 Decision Trace | 已实现 |
| 在线学习 | 固定规则难适应反馈，回复目标多会压倒发帖 | 动作类型分层 Softmax、冷启动 Action Prior、32 维 LinUCB | 代码及真实更新记录存在；模型持久化一致性待处理 |
| 内容质量 | 文本模板化、跑题、引用不清 | 独立的发帖/回复 Prompt、随机写作档案、来源选择、线程上下文、审核与修复 | 已实现；仍依赖模型语义判断 |
| 热点分支 | 热门话题需引用真实新闻，冷却后仍可能反复抢占发帖 | 历史 Trending 热度门槛、当前冷却扣分、动态入层概率及层内 Softmax；匹配当前快照新闻 | 新闻匹配与完整链接已实现，离线验证通过；独立层反馈不训练主策略 |

这些阶段是代码演进顺序，不代表每一步都已证明对比赛成绩有正向因果效果。相关设计记录见 [ADR](ADR.md)。

## 3. 总体架构与一轮运行

~~~mermaid
flowchart TD
    A["CTN SDK：Trending / Latest / News"] --> B["RawSnapshot + 原始 JSON"]
    A2["账户 Home / Tag / Notification"] --> B
    B --> C["Normalizer：Post / Article / Event"]
    C --> D[("SQLite snapshots")]
    D --> E["StateBuilder：历史热度 + 可选 Topic 分配"]
    T["TopicClassifier + Registry（可选）"] --> E
    E --> W["WorldState"]
    W --> F["StateEncoder：Global / Post / Event / Topic"]
    W --> G["候选生成与调度"]
    F --> H["战略评估 / 机会评估 / 启发式评分"]
    G --> I["DecisionEngine：Softmax 或 LinUCB"]
    H --> I
    J[("AgentState + 行动历史")] --> G
    J --> I
    I --> P["ActionPlan + 完整 Decision Trace"]
    P --> L["文本行动：LLM 生成、校验、审核、修复"]
    P --> X["非文本行动或 WAIT"]
    L --> X
    X --> K["Executor：SDK 写入或 Dry-run"]
    K --> Q[("结果日志 + 定时反馈窗口")]
    Q --> R["RewardEvaluator + Policy Update"]
    R --> I
~~~

Runtime 每轮先处理到期反馈，再采集新快照、重建状态、编码、装载 AgentState、选择行动、生成内容、执行和记录结果。**WorldState 描述外部环境；AgentState 描述代理自己做过什么。** WorldState 可根据数据库快照重建，AgentState 依赖行动日志而不是混入外部帖子事实。

主要入口是 [main.py](../main.py)、[agent/runtime.py](../agent/runtime.py)。默认 CLI 模式为 dry-run，默认策略为 bandit；平台写入必须显式使用 live 模式。

## 4. State：如何从原始观测构建环境状态

### 4.1 观测与标准化

基础 Observer 一次读取 Trending 帖子最多 15 条、Latest 帖子最多 30 条、新闻最多 20 篇，并记录开始与结束时间。Runtime 还尝试扩充 Home、热门标签和未处理通知来源。原始返回保存为 JSON；标准化结果以 snapshot_id 写入 SQLite。

~~~mermaid
flowchart LR
    R["RawSnapshot\n开始/结束时间 + 多路 Feed + 新闻"] --> N1["按 post_id 去重\n保留 feed_sources"]
    R --> N2["按 site:slug 去重文章"]
    N2 --> N3["相同 story_id 合并为 Event\n无 story_id 则一文一事件"]
    N1 --> S[("SQLite snapshots")]
    N3 --> S
    S --> H["寻找此前有效热度观测"]
    T["Topic Registry\n按 snapshot + 版本取分配"] --> B
    H --> B["StateBuilder"]
    S --> B
    B --> W["WorldState\nMetadata / Global / Post / Event / Topic"]
~~~

帖子去重键是 post_id；重复出现在多个 Feed 时，标准化器补充 feed_sources，其他字段保留首次观测值。文章键是网站子路径与文章 slug 的组合；同一 story_id 的多篇报道归入一个 Event，文章本身仍独立保存，便于溯源和防止新闻事件重复计数。代码见 [state/normalizer.py](../state/normalizer.py) 与 [state/store.py](../state/store.py)。

### 4.2 历史特征

对当前热度有数值的帖子，StateStore 只检索**本次观测开始前已完成**的历史快照，并选取最近一条同 post_id 且有热度值的观测。设当前热度为 \(h_t\)，上一条有效观测为 \(h_{t-1}\)，两次观测完成时间相隔 \(\Delta t\) 分钟：

\[
\Delta h_t=h_t-h_{t-1},\qquad
v_t=\frac{\Delta h_t}{\Delta t}.
\]

若缺少历史记录、任何热度值缺失或 \(\Delta t\leq0\)，hot_delta、hot_velocity 和 interval_minutes 保持 NULL；**未知热度变化与真实零增长不是一回事**。实现见 [state/features.py](../state/features.py) 和 [state/store.py](../state/store.py)。

### 4.3 可选的动态主题

Topic Classification 不使用人工固定主题表。Stage 1 在一个小批次中识别并合并相近议题；Stage 2 与历史 Topic Registry 比较，尽量复用已有 Topic ID，必要时新建。分类对象包括有正文的帖子与新闻事件，缓存按快照、文本哈希和分类器版本维护。Runtime 只有在启用 topics 且 LLM 可用时运行该流程，分类失败不会阻止构建 WorldState。

主题分配可以是 partial；WorldState.metadata 记录 eligible_posts、classified_posts、eligible_events、classified_events 及 topic_status。历史小模型试验暴露过误分类和语义重复，所以当前策略**没有直接使用 Topic 向量作为可信评分信号**。见 [state/topic_classification](../state/topic_classification) 与 [state/builder.py](../state/builder.py)。

### 4.4 WorldState 的对象边界

| 对象 | 核心内容 | 决策用途 |
|---|---|---|
| Metadata | snapshot_id、观测时间、分类版本与覆盖率 | 同步与数据完整性 |
| GlobalState | 帖子、Trending、Latest、文章、事件和主题统计 | 全局紧迫度、候选上下文 |
| PostState | post_id、作者、正文、热度、排名、Feed 来源、历史变化、Topic ID | 回复/互动候选、帖子特征 |
| EventState | event_id、关联文章 ID、Topic ID | 原创发帖候选 |
| TopicState | 主题与关联帖子/事件数量 | 描述与回放；当前未直接进入策略向量 |

由于 WorldState 是按 snapshot_id 重建的，重启后可以重新计算同一快照的状态和编码。当前 Topic Registry 的语义质量仍限制了主题特征的决策价值。

## 5. Encoder：State 如何变成策略输入

### 5.1 分层而非一个巨大固定向量

一轮快照的帖子和事件数量可变，编码器因此返回一个固定 12 维的 Global 向量，以及按对象 ID 索引的 Post 8 维、Event 1 维和 Topic 2 维向量。编码版本为 state-encoder-v1。

~~~mermaid
flowchart LR
    W["WorldState"] --> G["Global 向量\n12 维"]
    W --> P["Post 向量映射\npost_id → 8 维"]
    W --> E["Event 向量映射\nevent_id → 1 维"]
    W --> T["Topic 向量映射\ntopic_id → 2 维"]
    G --> C["候选上下文特征\n32 维"]
    P --> C
    E --> C
    T -. "目前仅描述，不直接输入" .-> C
    C --> D["LinUCB / 决策评分"]
~~~

基础缩放函数定义为：

\[
p_s(x)=\frac{\max(0,x)}{\max(0,x)+s},\qquad
q_s(x)=\frac{x}{|x|+s},\qquad
\rho(a,b)=
\begin{cases}
\mathrm{clip}(a/b,0,1),&b>0,\\
0,&b\leq0.
\end{cases}
\]

其中 \(p_s\) 压缩非负规模，\(q_s\) 保留速度正负方向。缺失、非有限数值及布尔值不当作有效数字；编码位置可置零，但另以 has_* 或 coverage 标记缺失，以免将未知误读为真实零。实现见 [state/encoder.py](../state/encoder.py)。

### 5.2 各对象的实际特征顺序

**Global 12 维**，设帖子数 \(N_p\)、文章数 \(N_a\)、事件数 \(N_e\)：

| 位置 | 特征 | 编码 |
|---:|---|---|
| 1 | post_volume | \(p_{50}(N_p)\) |
| 2 | trending_fraction | Trending 帖数 / \(N_p\) |
| 3 | latest_fraction | Latest 帖数 / \(N_p\) |
| 4 | article_volume | \(p_{30}(N_a)\) |
| 5 | event_volume | \(p_{20}(N_e)\) |
| 6 | mean_hot_score | 有热度帖的 \(p_{100}(h_i)\) 之均值 |
| 7 | max_hot_score | 有热度帖的 \(p_{100}(h_i)\) 之最大值 |
| 8 | hot_score_coverage | 有热度帖数 / \(N_p\) |
| 9 | mean_hot_velocity | 有速度帖的 \(q_2(v_i)\) 之均值 |
| 10 | velocity_coverage | 有速度帖数 / \(N_p\) |
| 11 | post_topic_coverage | 已分类有正文帖 / 有正文帖 |
| 12 | event_topic_coverage | 已分类事件 / 事件总数 |

第 6 维是 \(\operatorname{mean}(p_{100}(h_i))\)，不是 \(p_{100}(\operatorname{mean}(h_i))\)。

**单帖 8 维**：

\[
P_i=\big[p_{100}(h_i),\ I(h_i\text{存在}),\
I(\text{Trending}),\ I(\text{Latest}),\
1/\mathrm{rank}_i\text{ 或 }0,\ I(\mathrm{rank}_i>0),\
q_2(v_i),\ I(v_i\text{存在})\big].
\]

**单事件 1 维**：\(E_j=[p_3(\text{关联文章数})]\)。

**单主题 2 维**：\(T_k=[\rho(\text{主题关联帖数},N_p),\rho(\text{主题关联事件数},N_e)]\)。Topic 向量已生成，但目前的主策略不直接读取它。

### 5.3 喂给决策层的候选上下文

对每个候选 \(c\)，ContextFeatureEncoder 从全局向量中取 12 维，从目标帖子取 8 维（无目标则补零），从关联新闻事件取 1 维（无关联则补零），再拼接行动 one-hot 和策略评估量：

\[
x_c=[1,\ G_{12},\ P_{8},\ E_{1},\ \mathrm{onehot}_{6}(a),\
O_c,\ \mathrm{coverage}_c,\ U,\ \mathrm{risk}_c]\in\mathbb{R}^{32}.
\]

六类 one-hot 为 CREATE_POST、FOLLOW、LIKE、REPLY、REPOST、WAIT。MENTION 不在这个 one-hot 中，也不参加当前 LinUCB 训练。Topic ID、作者/NPC 身份、语气、写作档案和实际文本内容都没有进入这 32 维向量。因此模型目前学习的是**行动、环境和对象层的关联**，不能声称它直接学会了哪种语气或文本写法更有效。见 [policy/bandit.py](../policy/bandit.py)。

## 6. 决策层：候选、约束、评分与选择

### 6.1 候选行动和资格门槛

CandidateGenerator 根据 WorldState、EncodedWorldState、AgentState 和 PlatformState 生成候选。常规新闻发帖要求有标题、近期且与当前议题相关的报道；回复要求真实目标正文且目标仍在时间窗口内。LIKE、REPOST、FOLLOW 和 MENTION 需要明确的正向立场信号，热度本身不足以触发。WAIT 是显式候选。执行前再检查 feed lock、预算、账户已有关系、历史动作和重复内容。

| 行动 | 对象 | 是否用 LLM | 主要执行接口 |
|---|---|---|---|
| CREATE_POST | 常规新闻事件；或历史 Trending 话题匹配到的当前新闻 | 是 | account.post |
| REPLY | 具体帖子 | 是 | account.reply |
| LIKE | 具体帖子 | 否 | account.like |
| REPOST | 具体帖子 | 否 | account.repost |
| FOLLOW | 账户 | 否 | account.follow |
| MENTION | 在己方帖子中提及账户 | 否 | account.mention |
| WAIT | 无 | 否 | 不进行平台写入 |

ActionPlan 固定记录 action_id、snapshot_id、action_type、target/source、score、components、reason、parameters、生成内容和状态。action_id 由 snapshot_id 与候选键确定性生成；执行器先在 SQLite 预留动作，避免相同键的重复或结果不确定写入。见 [policy/models.py](../policy/models.py)、[policy/candidates.py](../policy/candidates.py)。

### 6.2 战略和机会评估

StrategicEvaluator 只计算可观测的紧迫度：

\[
U=\mathrm{clip}_{[0,1]}\left(
0.25\,\mathrm{trending\_fraction}
+0.75\,\max(0,\mathrm{mean\_hot\_velocity})
\cdot\mathrm{velocity\_coverage}\right).
\]

Favorability 保持 NULL；代码没有把公投民调、阵营身份或官方分数伪装成已观测状态。OpportunityEvaluator 的已观测项为：

\[
O_c=0.30\,\mathrm{influence}
+0.20\,\mathrm{freshness}
+0.15\,\mathrm{reach}
+0.20\,\mathrm{relevance},
\]

只累加可用项，并另存 evidence_coverage 与 missing_signals；不凭空补足战略契合度或比赛效果。见 [policy/strategic.py](../policy/strategic.py)、[policy/opportunity.py](../policy/opportunity.py)。

### 6.3 可解释启发式分数

默认设置下，非 WAIT 候选的启发式分数为：

\[
\begin{aligned}
h(c)=\mathrm{clip}_{[0,1]}(&p_a
+0.18r+0.14f+0.17H+0.09v\\
&+0.16e+0.12k+0.06q+b_{\mathrm{source}}
-0.20I_{\mathrm{repeat}}-0.08\,\mathrm{cost}).
\end{aligned}
\]

其中 \(p_a\) 是行动优先级：CREATE_POST 默认 \(+0.35\)，普通 REPLY 默认 \(-0.12\)，回复自己帖子的评论默认 \(+0.45\)；WAIT 基线为 0.22。各项分别表达相关性、新鲜度、热度、趋势、证据、完整性、预算和来源。随后乘以按行动类型配置或早期学习得到的 Action Prior：

\[
h'(c)=\mathrm{clip}_{[0,1]}\big(h(c)\,\pi_{\mathrm{type}(c)}\big).
\]

行动优先级与**事后真实 Reward 分开**：提高发帖优先级不会凭空给一次发帖更高的观测奖励。见 [policy/scorer.py](../policy/scorer.py)、[policy/settings.py](../policy/settings.py)。

冷启动或 heuristic 模式对非 WAIT 候选使用：

\[
s_{\mathrm{prior}}(c)=\mathrm{clip}_{[0,1]}
(0.75h'(c)+0.20O_c+0.05U),
\]

WAIT 仍取 0.22。这里的分数是选择用的估计量，不是成功概率。

### 6.4 上层约束与独立分支

~~~mermaid
flowchart TD
    C["已生成候选"] --> S["按内容槽/互动槽排除行动类型"]
    S --> R{"存在回复自己帖子的评论？"}
    R -- 是 --> R1["给该 REPLY 加优先分\n本轮延后蹭热度层"]
    R -- 否 --> H{"历史 Trending 话题匹配到当前新闻？"}
    H -- 是 --> B{"随机数 < p_hot？"}
    B -- 是 --> B1["进入蹭热度层\n按热度与冷却分数 Softmax 抽选"]
    B -- 否 --> N["普通候选池"]
    H -- 否 --> N
    R1 --> N
    N --> P{"最近成功发帖占比低于目标？"}
    P -- 是且有合格候选 --> P1["仅在合格常规发帖候选中选择"]
    P -- 否 --> Z["Softmax 选择"]
    P1 --> Z
    B1 --> X["ActionPlan\nexclude_from_reward=true"]
    Z --> Y["ActionPlan"]
~~~

- **最低原创覆盖**：最近最多 20 个成功文本行动中，\(C=N_{\mathrm{CREATE}}/(N_{\mathrm{CREATE}}+N_{\mathrm{REPLY}})\)。当 \(C<0.55\) 且存在合格、未被覆盖调度排除的常规发帖候选时，调度器把选择限制在这些发帖候选；这仍不改变 Reward。
- **自己帖子收到回复**：候选加较高的回复优先级，属于加分而非绝对硬选；存在此类回复时，本轮延后蹭热度层。
- **蹭热度**：StateStore 回看历史快照中曾进入 Trending Feed 的帖子，把它们作为新闻话题信号。候选要求该帖的历史或当前可见热度分数至少达到可配置门槛 `CTN_HOT_NEWS_SCORE_THRESHOLD`（默认 30），并能匹配**当前快照已采集的新闻**（Observer 默认最多取最新 20 篇）。帖龄 36 小时与当前是否仍在 Trending 不作硬性资格门槛；冷却会扣分。候选保留热门帖 ID 及热度证据供 Decision Trace 使用，发布内容依据匹配新闻并引用其完整原 URL，不附社交帖 URL。仍跳过自己的热门帖、已成功处理的同一热门帖，以及已引用过同一新闻 URL 的热点候选。若本轮存在回复自己帖子的候选，蹭热度层暂缓；否则按以下动态概率进入，层内按评分 Softmax 随机选择。该分支保留 `exclude_from_reward=true`，其反馈可观察，但不更新主策略。

设门槛为 \(T=30\)，历史和当前热度分别为 \(h_{\mathrm{hist}}\) 与 \(h_{\mathrm{now}}\)。仅当 \(\max(h_{\mathrm{hist}},h_{\mathrm{now}})\ge T\) 且有匹配新闻时才生成候选。当前仍在 Trending 时冷却量 \(q=0\)；当前无热度观测时 \(q=1\)；否则
\[
q=\max\left(0.6,\min\left(1,\frac{\max(h_{\mathrm{hist}},h_{\mathrm{now}})-h_{\mathrm{now}}}{\max(h_{\mathrm{hist}},h_{\mathrm{now}},1)}\right)\right).
\]
热度强度 \(h^*=\min(1,h_{\mathrm{now}}/\max(1,1.5T))\)，缺失当前热度时取 0。蹭热度候选的评分附加项为 \(\Delta s=0.14h^*-0.28q\)，总分仍按 PolicyScorer 规则截断到 \([0,1]\)。若候选集合为 \(C_H\)，则
\[
P(\text{进入蹭热度层})=0.5\left(1-0.8\min_{c\in C_H}q_c\right),\qquad
P(c\mid\text{已入层})=\operatorname{Softmax}(s_c/\tau).
\]
所以全部候选都完全冷却时入层概率为 0.1；仍有未冷却候选时最高为 0.5。热度加分、冷却扣分与入层概率属于**行动选择**，不进入 Observed Reward；温度与随机种子决定实际抽样。

当前实现先把热门帖中的新闻 URL 与本轮已观测文章 URL 规范化后精确匹配；热门帖没有可识别新闻 URL 时，才使用词项门槛。令 \(U(p)\) 为热门帖中可识别新闻 URL 的规范化集合，\(u(a)\) 为文章的规范化 URL；\(T(p)\) 为热门帖正文去通用词后的词集合，\(H(a)\) 和 \(S(a)\) 分别为当前快照新闻标题与摘要去通用词后的词集合：

\[
\operatorname{match}(p,a)=
\begin{cases}
1,& U(p)\ne\varnothing\ \land\ u(a)\in U(p),\\
1,& U(p)=\varnothing\ \land |T(p)\cap(H(a)\cup S(a))|\geq 2 \land\ |T(p)\cap H(a)|\geq 1,\\
0,& \text{其他情况}.
\end{cases}
\]

词项分支的两个交集门槛是保守过滤条件，不证明帖子和新闻语义等价。文章自身还须通过议题相关性检查，且标题、站点和 URL 有效。若热门帖含新闻链接但本轮新闻中没有同一链接，则不回退到词项匹配；两条路径均不额外联网抓取新闻。

- **新闻事件冷却**：常规发帖保留 event_cooldown_active 标记，用于最低覆盖调度的候选筛选；当前代码没有把它实现成严格两小时禁发，也没有在 PolicyScorer 中施加冷却惩罚。
- **非 LLM 互动频率**：Runtime 的第一槽选择文本行动或 WAIT；第二槽选择互动或 WAIT，且默认要求距上次成功互动已有至少 3 次成功文本行动。
- **WAIT**：由决策层选。内容 Prompt 不把 WAIT 当作特殊生成文本；连续内容 WAIT 达到两次时，Runtime 在可用候选范围内尝试排除 WAIT。

历史 Trending 的读取与缓存见 [state/store.py](../state/store.py) 和 [policy/candidates.py](../policy/candidates.py)；调度与分支见 [policy/engine.py](../policy/engine.py)。

## 7. 在线学习：LinUCB、Softmax 与参数更新

### 7.1 模型与选择公式

当前使用一个共享的 32 维 LinUCB 模型，岭正则初值 \(\lambda=1\)，探索系数 \(\alpha=0.15\)：

\[
A_0=\lambda I,\qquad b_0=0,\qquad
\hat\theta=A^{-1}b,
\]

\[
\mu_c=x_c^\top\hat\theta,\qquad
u_c=\sqrt{x_c^\top A^{-1}x_c},\qquad
\mathrm{UCB}_c=\mu_c+\alpha u_c.
\]

训练样本数至少为 5、且启动模式为 bandit 时，CREATE_POST/REPLY 的选择分数还加入行动优先级、Action Prior 调整、机会先验和风险项：

\[
s(c)=\mu_c+\alpha u_c+p_a+\Delta\pi_a
+0.10O_c-1.0\,\mathrm{risk}_c.
\]

WAIT 仍用固定基线；LIKE/REPOST/FOLLOW/MENTION 因缺少行动自身的可信结果，继续使用启发式分数。模型的预测值、探索奖励和最终选择分数分别保存，不能把探索 bonus 当成观测 Reward。

为了避免“回复对象很多，所以回复类型必然被选中”，选择分两层。设 \(M_a\) 为行动类型 \(a\) 下候选的最高分，默认温度 \(T=0.35\)：

\[
P(a)=\frac{\exp(M_a/T)}{\sum_b \exp(M_b/T)},\qquad
P(c\mid a)=\frac{\exp(s(c)/T)}
{\sum_{j:\mathrm{type}(j)=a}\exp(s(j)/T)},
\qquad
P(c)=P(a)P(c\mid a).
\]

温度越大，概率越平；实际动作由随机采样决定。蹭热度层在上述普通策略之外按第 6.4 节运行。见 [policy/bandit.py](../policy/bandit.py)、[policy/engine.py](../policy/engine.py)。

### 7.2 什么反馈会更新模型

成功 CREATE_POST/REPLY 的默认定时窗口为执行后 T+5 分钟、T+30 分钟、T+2 小时和 T+6 小时。只有所有计划窗口都不再 PENDING，Reward 才成为可用于训练的 FINAL；因此成功行动的训练通常至少延后约 6 小时。已知 FAILED/BLOCKED 文本行动可立即形成最终执行结果并尝试训练；其 Reward 不一定恰好为零，因为计算会使用可用组件。UNKNOWN 执行结果无有效奖励，不训练。WAIT、非文本互动、Dry-run、学习开启前的历史行动以及独立蹭热度层都不训练 LinUCB。

对一次合格的最终奖励 \(r\in[0,1]\)，模型更新为：

\[
A\leftarrow A+x_cx_c^\top,\qquad
b\leftarrow b+r x_c.
\]

实现为了避免每次重新求逆，使用 Sherman–Morrison 更新：

\[
A_{\mathrm{new}}^{-1}
=A^{-1}-\frac{A^{-1}x_cx_c^\top A^{-1}}
{1+x_c^\top A^{-1}x_c}.
\]

选中候选的 \(x_c\) 在决策时存入 candidate_scores，奖励到期后按 action_id 读取；同一行动有“至多一次更新”的记录检查，参数前后状态进入 policy_updates，并持久化到 policy_states。源码见 [agent/feedback.py](../agent/feedback.py)、[agent/v03_store.py](../agent/v03_store.py)。

### 7.3 冷启动 Action Prior 与学习开关

在模型前 5 个有效样本期间，除了 LinUCB，还对相应行动类型的 Action Prior 做小幅更新。默认学习率 \(\eta=0.05\)、每次均值更新最大 \(\pm0.01\)、向基线 0.5 回归的衰减 \(d=0.02\)、均值偏离不超过 0.20：

\[
m'_a=0.5+(m_a-0.5)(1-d),\quad
\delta=\mathrm{clip}\big(\eta(r-m'_a),-0.01,0.01\big),
\]

\[
m_a^{\mathrm{new}}
=\mathrm{clip}(m'_a+\delta,0.3,0.7),\qquad
\pi_a=0.5+m_a^{\mathrm{new}}\in[0.8,1.2].
\]

这限制了单条爆款内容对下一轮参数的推动幅度。代码默认 **CTN_ENABLE_LEARNING=false**；当前本机配置在审计时为 true。CLI 的 **--policy bandit** 会在模型样本数达到 5 后使用上下文 LinUCB；**--policy heuristic** 不使用 LinUCB 预测，但若 Action Prior 已学习并被 Runtime 刷新，仍可能受其影响。关闭学习开关控制的是后续参数更新；已持久化的 LinUCB 模型仍可能在 bandit 模式下被读取用于选择。见 [config.py](../config.py)、[policy/learning.py](../policy/learning.py)。

## 8. 大模型调用：Prompt、内容生成与质量控制

### 8.1 决策与写作的职责

策略层决定行动类型、对象、时机与写作档案；LLM 负责写文本，不直接决定是否行动。CREATE_POST 和 REPLY 分别使用独立的结构化 Prompt：

| 字段 | CREATE_POST | REPLY |
|---|---|---|
| task / goal | 针对 H2D2 议题中的新闻或讨论提出鲜明、具体的个人观点 | 针对目标帖的实际论点或直接问题作出回应 |
| tone / generation_profile | 策略后选中的语气、角度、结构、声音和强度 | 同左，但从回复专用档案选择 |
| target_post | 空 | 目标帖正文，最多 1200 字符 |
| thread_replies | 空 | 最近最多 3 条，每条正文最多 200 字符，附可用作者信息 |
| sources | 常规新闻，或与 Trending 话题匹配的当前新闻；热门帖仅作话题信号 | 默认不引入邻近新闻，减少跨话题漂移 |
| constraints | 观点鲜明、来源真实、表达自然、避免口号、字符预算 | 直答、上下文相关、事实可靠、避免套话、字符限制 |

当前写作模板回滚到用户此前选定的**直率讨论版**：发帖强调从新闻事实提出自己的观点与公共议题含义；回复强调识别原帖核心论点、直接回答问题并就论据展开讨论。模板保留账户公开的 Vote Yes 立场，但不要求每条内容重复投票声明；不再注入家庭、子女或父亲的固定人物背景。发帖和回复依然使用独立约束、相同 JSON 字段与变量注入接口；修复稿读取同一版约束，避免只在初稿中生效。事实与观点要区分，不能编造新闻、数字或个人经历。回滚前模板备份于 [prompt_templates_before_direct_rollback_2026-11-02.json](prompt_templates_before_direct_rollback_2026-11-02.json)。

写作档案每类行动各有 8 个，从最近历史中排除相同档案和相近结构后随机抽取；这增加表达变化，但**档案选择目前不由 LinUCB 学习**。另有约 30% 的程序概率附加“内容参与指南”，以及现有 tone、来源与线程变量注入。提示词回滚是可逆的内容实验；只读反馈数据稀疏且伴有 LLM 限流，不能据此证明近期互动下降由人物背景模板造成。实现见 [policy/generation_profiles.py](../policy/generation_profiles.py)、[agent/content_generator.py](../agent/content_generator.py)。

### 8.2 新闻来源选择与引用

常规新闻候选先由 CandidateGenerator 按议题相关性和 72 小时窗口筛选；同一事件下可带最多 3 篇候选文章。生成器不取第一篇有效文章，而对标题、网站和 URL 完整的候选进行确定性排序。其局部结构评分为：

\[
Q(d)=0.18\min(10,\text{标题中长度}\geq3\text{的不同词数})
+\frac{\min(1000,\text{摘要字符数})}{250}
+2\max\left(0,1-\frac{\text{报道年龄小时}}{168}\right),
\]

最后一项只在发布时间与观测时间可解析且报道年龄非负时计算。同分按标题、URL 和规范化数据稳定决胜。这个排序衡量**可引用材料的完整程度与时效**，不等于事实真伪或公众讨论价值的语义判定。结果写入 ActionPlan.parameters.source_selection。

CREATE_POST 会对新闻发帖追加来源名、标题、完整新闻 URL 和空间允许时最多三个合规热门标签。新闻 API 的 `url` 可能是 `/kingstonia-herald/articles/...` 相对路径，旧生成器曾原样发布，因此部分已发布帖子没有完整原文链接。现在选定新闻后按 SDK 基址 `https://sites.legitreal.com` 补全，并同步写入本次 ActionPlan 的新闻来源上下文；提示词、最终正文和本地来源校验使用同一完整 URL。蹭热度先用历史 Trending 帖子匹配当前快照新闻，再基于该新闻写原创观点。热门社交帖只作为选题信号和 Decision Trace 依据，正文不附其 URL 或 `Source: Legit Social` 来源行。生成层还会拒绝没有真实新闻来源的热点发帖，并在调用 LLM 前停止。离线用例已通过；尚未真实发布以验证平台链接预览。已发布帖子、原始快照和 SQLite 历史记录不会被追改。

若平台上限为 500 字符，先按实际要追加的后缀计算长度 \(L\)：常规与蹭热度发帖都计入所引新闻的完整 URL 与来源行，不计入热门社交帖 URL 或社交平台来源行。给模型的正文上限为：

\[
B=\min(350,\ 500-L).
\]

发帖标签与标签 Feed 采集分开：Runtime 仍只读取两条选定标签 Feed，以维持 API 调用量；发布时则从本轮平台热门标签中筛选较大的相关池。一般标签包括 H2D2、公投和 Kingstonia；支持立场的常用标签可轮换；护士、商户、地名等标签只有在候选新闻标题或摘要出现相应上下文时才进入池。反向、无关或未在本轮热门列表出现的标签不添加。抽样按热门帖数的对数加权，最近三条成功原创帖用过的标签按新近程度降权；先抽一条一般标签，再抽其余标签，最多三条。抽中的预选标签写入 content_brief，最终通过字符预算的标签写入 ActionPlan.selected_tags。该变化只影响内容后处理，不改变行动 Reward 或两个标签 Feed 的采集数量。

预算不足时先删标签；仍少于最短正文要求则在调用 LLM 前停止。模型若自行写出正文 hashtag，程序在格式检查前剔除并记录原稿与剔除标记；仅由现有后处理追加选中的热门标签。剔除后仍须通过长度、数字和语义审核。该兜底已通过离线用例。见 [agent/content_generator.py](../agent/content_generator.py)。

### 8.3 回复线程与审核一致性

Live REPLY 在写入前重新读取目标帖，并读取最多 3 条近期评论。生成器把目标帖、评论和可用作者标识放入 thread_replies；初稿、语义审核和修复使用相同的 1200 字目标上限与同一组评论。评论用于理解讨论，不把评论者的话误归给原帖作者。可用事实数字的本地检查也基于这份可见上下文。

文本按以下顺序处理；新闻引用、热点匹配和标签兜底已由离线测试覆盖：

~~~mermaid
flowchart LR
    A["ActionPlan + 来源/目标 + Profile"] --> B["LLM 初稿"]
    B --> C0["剔除模型自写 hashtag 并留痕"]
    C0 --> C["本地格式检查\n长度 / 占位词 / @ / # / 数字"]
    C --> D["本地立场与完整性检查"]
    D --> E["独立 LLM 语义审核\n相关性 / 事实依据 / 人类语气"]
    E -- 通过 --> F["常规/热点帖均附所引新闻原文链接；追加标签并执行"]
    C -- 可修复问题 --> R["一次约束修复"]
    E -- 退回 --> R
    R --> E2["再次格式与语义检查"]
    E2 -- 通过 --> F
    E2 -- 失败 --> X["记录失败并跳过"]
~~~

草稿、修复和审核的默认温度分别为 0.9、0.35、0.1；默认 max_tokens 分别为 220、180、110。官方 LLM 客户端用本地锁与跨进程文件锁，按同一 team code 保持至少 21 秒请求间隔，并对 429/暂时连接错误做有界重试。Runtime 对内容生成失败最多再发起一次新草稿；平台对文本返回确定性的 400/413/422 时可再生成并提交一次。LLM 的 429 表示请求限流，不是草稿质量失败；客户端三次有界尝试仍失败后，生成器抛出 `LLMRateLimitError`，Runtime 记录失败但不在同轮重做完整生成。该分支已离线测试；外部额度不足或其他进程共享 team code 时，后续轮次仍可能遇到 429。语义审核降低跑题风险，但不能证明所有自然语言事实都正确。见 [state/topic_classification/client.py](../state/topic_classification/client.py)、[agent/runtime.py](../agent/runtime.py)。

本轮还核对了一条已发布的 CREATE_POST：它把报道中“缺少 sunset clauses 可能导致长期紧急状态”的批评写成了账户自身观点，却未提出与既定 Vote Yes 立场相符的论证；语义审核仍判通过。现有硬校验主要拦截明确的第一人称 Vote No 表态。已调整 CREATE_POST 的生成焦点和语义审核指令，要求以**最终草稿本身**判断立场：允许承认反方关切，但若草稿采纳反方结论且没有对应的 Yes 论证，则要求修订；不依赖特定报道关键词，也不强制每帖喊口号。尚未用官方 LLM 对该历史案例重新实测，不能保证零误判。

## 9. 执行、反馈、Reward 与可观测性

### 9.1 执行与幂等

Dry-run 生成和校验 ActionPlan，但不会调用平台写接口。Live 模式先检查 feed lock，再在 agent_actions 预留行动键，最后调用相应 SDK 方法。网络超时或文本写入后的 5xx 可能意味着远端已成功但响应丢失，记录 UNKNOWN 并阻断盲目重发；明确的 429 仅在满足等待边界时重试一次。执行结果与远端 post_id 写回同一行动日志。实现见 [agent/executor.py](../agent/executor.py)、[agent/history.py](../agent/history.py)。

### 9.2 定时反馈与归因边界

v0.3 FeedbackSystem 以 action_id 和 remote_id 为主键，在实际执行时间后安排 T+5 分钟、T+30 分钟、T+2 小时、T+6 小时窗口；数据库约束确保同一行动同一窗口不重复安排。Runtime 每轮开始处理到期窗口。

- CREATE_POST 读取自己帖子的公开浏览、点赞、转发、回复等计数。
- REPLY 分别读取**回复自身**与**父帖整体**计数，并在回复前尽可能保存父帖基线。父帖变化只供离线分析，标记 NOT_ATTRIBUTED_TO_REPLY，不计入回复 Reward。
- LIKE、REPOST、FOLLOW、MENTION 若无行动自身计数端点，窗口标记 UNAVAILABLE；缺少的指标保留 NULL。
- 独立的 feedback_monitor.py/observation_monitor.py 还有按帖子年龄采样的只读曲线系统，其 1m/5m/15m/30m/1h/3h/6h/24h 窗口与 v0.3 奖励窗口不是同一套。

状态 FINAL 只表示所有计划窗口都已离开 PENDING；其中仍可能有 PARTIAL、UNAVAILABLE 或 FAILED。系统能画**观察到的互动变化曲线**，但没有随机对照或反事实，不能把父帖或内容时机与增长直接解释为因果效果。

### 9.3 Reward v2 的实际公式

当前主流程使用 reward-v2-action-feedback。对最后一个反馈窗口，优先取行动自身的**区间增量**；增量字典为空时回退该窗口的绝对计数。令自身浏览、点赞、转发、回复计数分别为 \(V,L,S,R\)：

\[
P=V+5L+10S+15R,\qquad
F_{\mathrm{own}}=1-e^{-P/50}.
\]

可用组件包括执行成功 \(E\)、平台即时接受 \(C\)、本地来源词汇一致性 \(Q\)、回复与父帖的词项 Jaccard 相关性 \(J\)、自身互动 \(F_{\mathrm{own}}\)。适用且非 NULL 的分量按权重重归一化：

\[
R_{\mathrm{obs}}=
\frac{\sum_{k\in K_{\mathrm{known}}}w_k z_k}
{\sum_{k\in K_{\mathrm{known}}}w_k},
\quad
(w_E,w_C,w_Q,w_J,w_F)=(0.30,0.15,0.35,0.15,0.15).
\]

Q 只在有可匹配来源时适用，最高 0.5；它是 URL 匹配、词项重叠和有限数字冲突检查，不是完整事实核查。J 主要用于回复。新颖度采用正文规范化、精确哈希与 3 词片段 Jaccard，存入内容评估，但**当前不进入 Reward 合成**。没有可比历史时为 UNKNOWN/NULL，而不是给高新颖度。父帖计数也不进入 \(F_{\mathrm{own}}\)。因此 \(R_{\mathrm{obs}}\) 是局部质量与互动代理值，不是官方分数，也不能识别行动内容、对象和时机的因果贡献。见 [policy/reward.py](../policy/reward.py)、[policy/source_validator.py](../policy/source_validator.py)、[policy/novelty.py](../policy/novelty.py)。

### 9.4 完整 Decision Trace 与离线分析

每次决策的 trace 记录 snapshot、编码/策略版本、候选列表、基础分、Action Prior、机会分、缺失信号、32 维策略特征、预测值、探索 bonus、选择分数、选择概率、调度排除原因、最终选中项及写作档案。v0.3 SQLite 主要表如下：

| 表 | 保存内容 |
|---|---|
| snapshots | 标准化环境快照 |
| agent_actions | 行动计划、执行状态、远端 ID、参数与结果 |
| policy_decisions / candidate_scores | 完整决策与每个被评估候选的分数、概率和特征 |
| content_assessments | 内容结构、新颖度、来源匹配和回复相关性 |
| action_feedback_snapshots | 按窗口采到的自身及父帖计数、增量、状态 |
| action_rewards | 奖励版本、组件、计算依据、PENDING/PROVISIONAL/FINAL/INVALID |
| policy_updates / policy_states | 每次参数更新前后值与当前持久化模型 |

决策时 candidate_scores 中的内容新颖度与来源质量列尚未得知；它们可能在**选中行动生成内容之后**回填。因此不能把生成后的内容评估写成当轮策略已使用的输入。离线报告可按行动类型、发表时段、曝光和内容结构分层比较相关性；它们仍是观察性分析。见 [agent/v03_store.py](../agent/v03_store.py)、[analytics/policy_report.py](../analytics/policy_report.py)、[analyze_feedback.py](../analyze_feedback.py)。

## 10. 最终运行证据与结果复核

### 10.1 数据源、时间和统计口径

最终结果使用两类互相独立的证据。平台活动导出中的 `SCORE_NARRATIVE` 行是**官方计分记录**；本地 SQLite 的 snapshots、agent_actions、policy_decisions、action_feedback_snapshots 和 policy_updates 是**工程运行记录**。前者按批次汇总 NPC 互动，后者记录动作与内部反馈，两者没有可靠的逐条对应键。只将 16 条计分行整理为可分享的 [final_scoring_events.csv](data/final_scoring_events.csv)，保留每批发生时间、分数和可读出的计数；不把其他事件类型混入成绩。

官方计分记录的北京时间范围是 **2026-11-01 23:53 至 2026-11-02 13:46**。本地行动分析固定使用 `created_at < 2026-11-02 05:14:00 UTC` 的行动队列，对应北京时间 13:14 前创建的计划；此队列关联的后续反馈和参数更新仍从最终数据库读取。数据库中满足该截点的快照共 **1024** 份，最后一份为 snapshot 1025，含 128 条标准化帖子、20 篇文章和 14 个新闻事件。官方计分窗口与数据库行动队列起点不同，**不能用 1311 分除以全部行动数来估计单次行动收益**。

### 10.2 官方计分结果

| 指标 | 活动导出中的数量 | 正确解释 |
|---|---:|---|
| 计分批次 | 16 | 每条是平台汇总事件，不是一条 Agent 行动 |
| 合计得分 | **1311** | 16 条计分增量求和 |
| 计分浏览 | 123 | 活动日志列出的浏览，不代表全部曝光或独立人数 |
| 点赞 / 回复 / 关注 | 14 / 7 / 1 | NPC engagement 批次中明确列出的次数 |
| 朝本方的观点变化贡献 | 2 | 贡献记录；不能表述为“说服了 2 位选民” |
| 朝中立方向的观点变化贡献 | 5 | 与本方贡献分开统计 |

\[
S_{\mathrm{official}}=\sum_{j=1}^{16}\Delta S_j=1311.
\]

以下曲线由 16 条原始计分增量逐条累加而成；完整时间点及各类计数见上述衍生 CSV。曲线显示**计分入账时间**，不等同于内容被阅读或观点发生变化的精确时间。

~~~mermaid
xychart-beta
    title "官方活动日志累计得分（北京时间，2026-11-01 至 2026-11-02）"
    x-axis ["23:53", "00:53", "01:53", "02:53", "03:53", "04:53", "05:53", "06:53", "07:53", "08:20", "08:46", "09:46", "10:46", "11:46", "12:46", "13:46"]
    y-axis "累计分数" 0 --> 1400
    line [52, 170, 314, 354, 494, 694, 726, 742, 894, 918, 922, 1055, 1103, 1151, 1303, 1311]
~~~

### 10.3 本地行动、反馈与学习证据

截至上述行动截点，数据库记录了 **1272 次决策**与 **33181 行候选评分**。这说明策略实际评估过多种候选；候选行不是执行次数。成功文本行动为 **43 条 CREATE_POST** 和 **349 条 REPLY**，因此原创发帖在成功文本行动中的占比为 \(43/(43+349)=10.97\%\)。该占比是**成功执行后的分布**，并非策略的原始选择概率；发帖和回复分别另有 46 与 235 次 FAILED 记录，故真实输出受到生成和执行成功率影响。非文本行动中，LIKE、REPOST、FOLLOW、MENTION 分别有 163、128、144、136 条 SUCCEEDED 记录；WAIT 记录 356 条。这里的行动计数与官方 NPC 计分互不等价。

决策方式的归档分布为 heuristic_softmax 645 次、linucb_softmax 342 次、早期 heuristic 256 次、prior_softmax 16 次，以及热点独立层的旧随机选择 11 次和新版 Softmax 选择 2 次。不同名称反映同一项目在时间中的版本演进；不能将所有成绩归因于最后一个版本。

| 内部反馈窗口 | 已采到自身计数 | 不可用 | 仍待采集 | 说明 |
|---|---:|---:|---:|---|
| T+5 分钟 | 339 | 405 | 0 | 以行动自身远端 ID 为观察目标 |
| T+30 分钟 | 339 | 405 | 0 | 与上一窗口同一行动队列 |
| T+2 小时 | 339 | 405 | 0 | 只读公开互动计数 |
| T+6 小时 | 199 | 361 | 184 | 缺失和待采集均保留状态，不填零 |

表中各行是**窗口记录数**，不是彼此独立的行动数；不可用的窗口没有被解释为零互动。最近一次 `reward-v2-action-feedback` 的 FINAL 状态中，成功发帖 17 条的内部平均 Reward 为 0.750，成功回复 180 条为 0.629。它们是不同样本上、按代码定义的质量与自身反馈代理值，不能直接比较两种内容对比赛得分的因果作用。

关联本队列行动的 `policy_updates` 中，**244 条记录标记为已应用更新**；最终持久化 LinUCB 模型的 `count` 为 **209**。两者不是可以互换的样本数。策略确实曾以 linucb_softmax 选择行动并写入更新，但计数差异和已有的样本计数回退记录说明：在缺少事务化版本检查的情况下，不能声称学习状态形成了完全连续、单调的训练链。

### 10.4 验证范围与解释边界

本项目已有 test_encoder.py、test_worldstate.py、test_v02.py、test_v03.py、test_learning_policy.py 等表示、行动、反馈与学习测试。最后一轮相关离线套件覆盖 `test_policy`、`test_content_regression`、`test_v02`、`test_learning_policy`、`test_v03`、`test_action_policy`、`test_llm_client`，合计 **88 项测试通过**；另验证了三标签成文、字符预算、标签过滤与行动历史恢复。测试证明代码路径满足这些断言，不证明语义质量、线上链接预览或积分提升。

最终代码含有若干运行期间持续迭代的约束与提示词版本，活动导出又只提供按批汇总的成绩。项目没有按版本随机分组的对照实验，也没有官方逐行动说服结果。因此只能报告“观察到 1311 分及上述互动”，不能声称哪个 Prompt、发帖比例、热点分支、标签或在线学习更新创造了这些分数。Topic 分类语义仍依赖 LLM；内部 Reward 的反馈覆盖不完整；LinUCB 输入没有写作档案与文本语义；这些限制均应保留在面试讲解中。

## 11. 如何离线复现

在 CTN-bot 项目根目录使用虚拟环境。以下命令读取本地代码与 SQLite，不需要平台写入；`data/state.db` 与 `docs/data/final_scoring_events.csv` 保留各自的数据来源和口径。

~~~powershell
.\.venv\Scripts\python.exe -B -m unittest test_policy test_content_regression test_v02 test_learning_policy test_v03 test_action_policy test_llm_client
.\.venv\Scripts\python.exe -B learning_trace.py --hours 10000 --limit 20
.\.venv\Scripts\python.exe -B -m analytics.policy_report --db data\state.db
.\.venv\Scripts\python.exe -B analyze_feedback.py --db data\state.db
~~~

`main.py` 是历史运行入口；`--policy heuristic` 使用启发式 Softmax，`--policy bandit` 在满足样本门槛后使用上下文 LinUCB。`CTN_ENABLE_LEARNING` 决定后续是否写入参数更新。上述离线报告应与第 10 节的固定时间截点区分：默认命令可能扫描整库或按当前时间回看，数值不一定与本节归档队列完全相同。完整的状态、策略和内容执行路径见第 3–9 节。

## 12. 核心代码导航

| 层 | 主要文件 |
|---|---|
| 观测、标准化、快照 | [state/observer.py](../state/observer.py)、[state/normalizer.py](../state/normalizer.py)、[state/store.py](../state/store.py) |
| Topic、WorldState、编码 | [state/topic_classification](../state/topic_classification)、[state/builder.py](../state/builder.py)、[state/encoder.py](../state/encoder.py) |
| 候选、评分、选择、学习 | [policy/candidates.py](../policy/candidates.py)、[policy/scorer.py](../policy/scorer.py)、[policy/engine.py](../policy/engine.py)、[policy/bandit.py](../policy/bandit.py)、[policy/learning.py](../policy/learning.py) |
| 提示词与内容 | [agent/content_generator.py](../agent/content_generator.py)、[policy/generation_profiles.py](../policy/generation_profiles.py) |
| 执行与反馈 | [agent/runtime.py](../agent/runtime.py)、[agent/executor.py](../agent/executor.py)、[agent/feedback.py](../agent/feedback.py)、[policy/reward.py](../policy/reward.py) |
| 日志与报告 | [agent/history.py](../agent/history.py)、[agent/v03_store.py](../agent/v03_store.py)、[learning_trace.py](../learning_trace.py)、[analytics/policy_report.py](../analytics/policy_report.py) |

---

**结论**：本项目实现了环境观测、可重建 WorldState、分层编码、受约束候选策略、LLM 文本生成、平台执行、分窗口反馈和在线参数更新的完整工程链路。最终活动导出累计 **1311 分**，本地记录证实了真实行动、候选评分和 244 条已应用更新记录。工程贡献是可追溯的决策过程、对缺失反馈的显式建模，以及把内部 Reward 与官方成绩分开复核。现有证据不足以证明某次策略更新或某种内容写法带来了增量得分；持久化学习状态的一致性和对照评估仍是后续研究问题。

