---
layout: center
highlighter: shiki
css: unocss
colorSchema: dark
transition: fade-out
title: MedFusion - Promo MVP Deck
exportFilename: medfusion-promo-mvp-deck
lineNumbers: false
drawings:
  persist: false
mdc: true
clicks: 0
preload: false
routerMode: hash
---

<div class="text-center px-18">
  <div class="uppercase tracking-[0.35em] text-sky-300 text-sm mb-6">
    MedFusion Promo MVP
  </div>
  <h1 class="leading-tight">
    给导师和医生演示医学 AI<br />
    不该再从终端开始
  </h1>
  <div class="mt-6 text-xl text-zinc-300">
    小红书 / B 站口播版
  </div>
  <div class="mt-3 text-sm text-zinc-400">
    演示型 MVP，不夸大为完整生产平台
  </div>
</div>

---
layout: two-cols
---

# 这一版要解决什么

<div class="mt-8 space-y-4">
  <div class="rounded-xl border border-sky-500/30 bg-sky-500/10 p-5">
    <div class="text-sky-300 font-semibold mb-2">先让人看懂</div>
    <div class="text-sm text-zinc-300">
      数据集、训练过程、结果页和报告能被非工程用户理解。
    </div>
  </div>
  <div class="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-5">
    <div class="text-emerald-300 font-semibold mb-2">先让演示稳定</div>
    <div class="text-sm text-zinc-300">
      打开页面后不露怯，录屏时不会点到明显空壳。
    </div>
  </div>
  <div class="rounded-xl border border-amber-500/30 bg-amber-500/10 p-5">
    <div class="text-amber-300 font-semibold mb-2">先让结果能截图</div>
    <div class="text-sm text-zinc-300">
      ROC、混淆矩阵、注意力图、validation 摘要必须能拿出来展示。
    </div>
  </div>
</div>

::right::

# 这一版不做什么

<div class="mt-8 space-y-3 text-sm">
  <div class="rounded-xl border border-rose-500/20 bg-rose-500/10 p-4">
    不吹成“全功能多模态平台”
  </div>
  <div class="rounded-xl border border-rose-500/20 bg-rose-500/10 p-4">
    不在公开内容里碰复杂工作流编排
  </div>
  <div class="rounded-xl border border-rose-500/20 bg-rose-500/10 p-4">
    不把 demo 感很重的伪数据当成真实交付能力
  </div>
  <div class="rounded-xl border border-rose-500/20 bg-rose-500/10 p-4">
    不承诺已经生产可用
  </div>
</div>

---

# 演示闭环只有四步

<div class="grid grid-cols-4 gap-4 mt-12">
  <div class="rounded-2xl border border-zinc-700 bg-zinc-900/70 p-5">
    <div class="text-zinc-400 text-xs uppercase tracking-widest">Step 1</div>
    <div class="text-xl font-semibold mt-3">注册数据集</div>
    <div class="text-sm text-zinc-400 mt-3">
      把本地目录、样本数、类别数和说明先放到台面上。
    </div>
  </div>
  <div class="rounded-2xl border border-zinc-700 bg-zinc-900/70 p-5">
    <div class="text-zinc-400 text-xs uppercase tracking-widest">Step 2</div>
    <div class="text-xl font-semibold mt-3">发起训练</div>
    <div class="text-sm text-zinc-400 mt-3">
      用一套最小配置直接启动任务，而不是讲半天命令行。
    </div>
  </div>
  <div class="rounded-2xl border border-zinc-700 bg-zinc-900/70 p-5">
    <div class="text-zinc-400 text-xs uppercase tracking-widest">Step 3</div>
    <div class="text-xl font-semibold mt-3">看训练过程</div>
    <div class="text-sm text-zinc-400 mt-3">
      让观众看到进度、epoch、loss、accuracy 在推进。
    </div>
  </div>
  <div class="rounded-2xl border border-zinc-700 bg-zinc-900/70 p-5">
    <div class="text-zinc-400 text-xs uppercase tracking-widest">Step 4</div>
    <div class="text-xl font-semibold mt-3">看结果产物</div>
    <div class="text-sm text-zinc-400 mt-3">
      模型记录、图表、validation 和报告自动沉淀。
    </div>
  </div>
</div>

---
layout: two-cols
---

# 现在的结果页长这样

<div class="mt-6 text-sm text-zinc-300 leading-7">
  重点不是“训练结束”四个字，而是把结果页做成真的能讲、能截图、能转化的页面。
</div>

<div class="mt-6 space-y-3 text-sm">
  <div class="rounded-xl border border-sky-500/25 bg-sky-500/10 p-4">
    指标卡：AUC、宏平均 F1、Balanced Accuracy、Specificity
  </div>
  <div class="rounded-xl border border-sky-500/25 bg-sky-500/10 p-4">
    图表区：ROC、训练曲线、混淆矩阵、归一化混淆矩阵
  </div>
  <div class="rounded-xl border border-sky-500/25 bg-sky-500/10 p-4">
    Validation：per-class、阈值分析、ECE、Brier Score、误分类摘要
  </div>
  <div class="rounded-xl border border-sky-500/25 bg-sky-500/10 p-4">
    多模态展示：注意力图、注意力统计图、结果文件下载
  </div>
</div>

::right::

<div class="h-full flex items-center justify-center">
  <img
    src="./assets/promo-demo/model-library-page.png"
    class="max-h-[520px] rounded-2xl border border-zinc-700 shadow-2xl"
  />
</div>

---

# 这批图已经可以直接拿去做内容

<div class="grid grid-cols-[1.3fr_0.7fr] gap-8 mt-8 items-center">
  <div>
    <img
      src="./assets/promo-demo/showcase-grid.png"
      class="rounded-2xl border border-zinc-700 shadow-2xl"
    />
  </div>
  <div class="space-y-4 text-sm">
    <div class="rounded-xl border border-zinc-700 bg-zinc-900/70 p-4">
      README 可以直接展示训练曲线、ROC、归一化混淆矩阵和注意力图。
    </div>
    <div class="rounded-xl border border-zinc-700 bg-zinc-900/70 p-4">
      小红书更适合用“结果页 + 一张拼图”做第一眼吸引。
    </div>
    <div class="rounded-xl border border-zinc-700 bg-zinc-900/70 p-4">
      B 站更适合把 validation、artifact、报告链路讲完整。
    </div>
  </div>
</div>

---

# 没有私有数据也能先试

<div class="grid grid-cols-3 gap-5 mt-10">
  <div class="rounded-2xl border border-emerald-500/25 bg-emerald-500/10 p-5">
    <div class="text-sm uppercase tracking-widest text-emerald-300">最快开始</div>
    <div class="text-2xl font-semibold mt-3">MedMNIST</div>
    <div class="text-sm text-zinc-300 mt-4">
      最适合验证训练、结果页和报告链路。
    </div>
  </div>
  <div class="rounded-2xl border border-amber-500/25 bg-amber-500/10 p-5">
    <div class="text-sm uppercase tracking-widest text-amber-300">表格任务</div>
    <div class="text-2xl font-semibold mt-3">UCI Heart Disease</div>
    <div class="text-sm text-zinc-300 mt-4">
      先把 tabular 主链和指标输出跑通。
    </div>
  </div>
  <div class="rounded-2xl border border-sky-500/25 bg-sky-500/10 p-5">
    <div class="text-sm uppercase tracking-widest text-sky-300">真实医学影像</div>
    <div class="text-2xl font-semibold mt-3">ISIC / HAM10000 / ChestXray14</div>
    <div class="text-sm text-zinc-300 mt-4">
      更适合对外展示和后续做论文风格验证。
    </div>
  </div>
</div>

<div class="mt-8 text-sm text-zinc-400">
  重点不是“收很多数据集”，而是先给内容用户一个看完就能自己试的入口。
</div>

---
layout: two-cols
---

# 传播线和交付线要一起讲清楚

<div class="mt-8 space-y-4">
  <div class="rounded-2xl border border-sky-500/25 bg-sky-500/10 p-5">
    <div class="text-xl font-semibold text-sky-300">传播展示线</div>
    <div class="text-sm text-zinc-300 mt-3">
      负责让导师、医生、研究生第一眼看懂，愿意停留和咨询。
    </div>
  </div>
  <div class="rounded-2xl border border-emerald-500/25 bg-emerald-500/10 p-5">
    <div class="text-xl font-semibold text-emerald-300">实用交付线</div>
    <div class="text-sm text-zinc-300 mt-3">
      负责真实训练、artifact、validation、报告和复现能力。
    </div>
  </div>
</div>

::right::

# 后续亮点怎么讲

<div class="mt-8 space-y-3 text-sm">
  <div class="rounded-xl border border-zinc-700 bg-zinc-900/70 p-4">
    现在主打：数据集 -> 训练 -> 结果 -> 报告
  </div>
  <div class="rounded-xl border border-zinc-700 bg-zinc-900/70 p-4">
    下一步主打：公开数据集快速验证
  </div>
  <div class="rounded-xl border border-zinc-700 bg-zinc-900/70 p-4">
    未来亮点：参考 ComfyUI 的节点式拖拽搭建器
  </div>
  <div class="rounded-xl border border-zinc-700 bg-zinc-900/70 p-4">
    但要明确：节点式搭建是 roadmap，不是假装已经做完
  </div>
</div>

---

# 同一套内容，两个平台两种说法

<div class="grid grid-cols-2 gap-8 mt-10">
  <div class="rounded-2xl border border-rose-500/25 bg-rose-500/10 p-6">
    <div class="text-2xl font-semibold mb-4">小红书</div>
    <div class="space-y-3 text-sm text-zinc-300">
      <div>时长：60 到 90 秒</div>
      <div>重点：画面舒服、第一句抓人、结果页好看</div>
      <div>结构：痛点 -> 演示闭环 -> 结果页 -> CTA</div>
      <div>更适合用“导师汇报、组会展示、项目介绍”这种表达</div>
    </div>
  </div>
  <div class="rounded-2xl border border-sky-500/25 bg-sky-500/10 p-6">
    <div class="text-2xl font-semibold mb-4">B 站</div>
    <div class="space-y-3 text-sm text-zinc-300">
      <div>时长：3 到 5 分钟起步</div>
      <div>重点：把 validation、artifact、公开数据集入口讲清楚</div>
      <div>结构：背景 -> 产品思路 -> Demo -> 技术可信度 -> roadmap</div>
      <div>更适合顺带讲“为什么我不想再只用命令行演示”</div>
    </div>
  </div>
</div>

---
layout: center
---

<div class="text-center px-18">
  <div class="text-sm uppercase tracking-[0.35em] text-sky-300 mb-5">
    Closing
  </div>
  <h1 class="leading-tight">
    先把结果讲清楚<br />
    再把能力做扎实
  </h1>
  <div class="mt-6 text-lg text-zinc-300">
    MedFusion 当前更适合被定义为一套适合演示、教学、沟通和早期项目展示的医学 AI Web 控制台。
  </div>
  <div class="mt-8 text-sm text-zinc-400">
    后续内容可以继续围绕公开数据集验证、结果页强化和节点式搭建器 roadmap 展开。
  </div>
</div>
