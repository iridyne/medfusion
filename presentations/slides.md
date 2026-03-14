---
layout: center
highlighter: shiki
css: unocss
colorSchema: dark
transition: fade-out
title: MedFusion - 医学多模态深度学习框架
exportFilename: MedFusion - Medical Multimodal Deep Learning Framework
lineNumbers: false
drawings:
  persist: false
mdc: true
clicks: 0
preload: false
glowSeed: 233
glowPreset: cyan
routerMode: hash
---

<div translate-x--20 translate-y-10>

<h1>
MedFusion<br />医学多模态深度学习研究框架
</h1>

Medical Multimodal Deep Learning Research Framework

</div>

---
layout: intro
class: px-35
glowSeed: 128
glowPreset: blue
---

<div flex items-center gap-3>
  <div
    v-click="1"
    :class="$clicks < 1 ? 'translate-x--5 opacity-0' : 'translate-x-0 opacity-100'"
    flex flex-col items-start transition duration-500 ease-in-out min-w-60
  >
    <div text-6xl mb-5>🏥</div>
    <span font-semibold text-3xl>MedFusion</span>
    <div>
      <div>
        <span class="opacity-70">Modular Medical AI Framework</span>
      </div>
      <div text-sm flex items-center gap-2 mt-4>
        <div i-carbon:logo-github /><span underline decoration-dashed font-mono decoration-zinc-300>medfusion</span>
      </div>
    </div>
  </div>
  <div flex-1 />
  <div flex flex-col gap-8>
    <div mb-4 v-click="2">
      <div mb-4 text-zinc-400>
        <span>核心技术</span>
      </div>
      <div
        flex flex-wrap items-start content-start gap-4 transition duration-500 ease-in-out
        :class="$clicks < 2 ? 'translate-y-5' : 'translate-y-0'"
      >
        <div flex items-center gap-2 text-2xl w-fit h-fit>
          <div i-logos:pytorch-icon inline-block /> PyTorch
        </div>
        <div flex items-center gap-2 text-2xl w-fit h-fit>
          <div i-logos:python inline-block /> Python
        </div>
        <div flex items-center gap-2 text-2xl w-fit h-fit>
          <div i-logos:react inline-block /> React
        </div>
        <div flex items-center gap-2 text-2xl w-fit h-fit>
          <div i-simple-icons:fastapi inline-block text-teal-400 /> FastAPI
        </div>
      </div>
    </div>
    <div v-click="3">
      <div mb-4 text-zinc-400>
        <span>支持场景</span>
      </div>
      <div
        flex flex-wrap items-start content-start gap-4 transition duration-500 ease-in-out
        :class="$clicks < 3 ? 'translate-y-5' : 'translate-y-0'"
      >
        <div flex items-center gap-2 text-xl w-fit h-fit>
          <div>🔬</div><div>多模态融合</div>
        </div>
        <div flex items-center gap-2 text-xl w-fit h-fit>
          <div>🏥</div><div>医学影像</div>
        </div>
        <div flex items-center gap-2 text-xl w-fit h-fit>
          <div>📊</div><div>生存分析</div>
        </div>
        <div flex items-center gap-2 text-xl w-fit h-fit>
          <div>🎯</div><div>MIL 聚合</div>
        </div>
      </div>
    </div>
  </div>
</div>

---
glowSeed: 15
glow: top
glowPreset: rust
---

# 医学 AI 研究的挑战

<div class="grid grid-cols-2 gap-4 mt-8">

<div v-click="1" class="bg-red-500/10 border border-red-500/30 rounded-lg p-6 transition duration-300" :class="$clicks < 1 ? 'translate-y-5 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-red-400 font-bold mb-2">🔴 架构复杂</div>
<div class="text-sm opacity-80">
多模态融合需要手动编写大量胶水代码
</div>
</div>

<div v-click="2" class="bg-red-500/10 border border-red-500/30 rounded-lg p-6 transition duration-300" :class="$clicks < 2 ? 'translate-y-5 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-red-400 font-bold mb-2">🔴 实验效率低</div>
<div class="text-sm opacity-80">
切换不同骨干网络需要修改代码重新调试
</div>
</div>

<div v-click="3" class="bg-red-500/10 border border-red-500/30 rounded-lg p-6 transition duration-300" :class="$clicks < 3 ? 'translate-y-5 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-red-400 font-bold mb-2">🔴 可复现性差</div>
<div class="text-sm opacity-80">
缺乏标准化配置系统，实验难以复现
</div>
</div>

<div v-click="4" class="bg-red-500/10 border border-red-500/30 rounded-lg p-6 transition duration-300" :class="$clicks < 4 ? 'translate-y-5 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-red-400 font-bold mb-2">🔴 医学场景特殊</div>
<div class="text-sm opacity-80">
多角度 CT、时间序列等场景缺乏统一支持
</div>
</div>

</div>

---
glowSeed: 88
glow: bottom
glowPreset: cyan
---

# MedFusion 解决方案

<div class="grid grid-cols-2 gap-4 mt-8">

<div v-click="1" class="bg-green-500/10 border border-green-500/30 rounded-lg p-6 transition duration-300 hover:scale-105" :class="$clicks < 1 ? 'translate-y-5 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-green-400 font-bold mb-2">✅ 完全解耦</div>
<div class="text-sm opacity-80">
29+ 骨干网络 × 5+ 融合策略 = 350+ 组合
</div>
</div>

<div v-click="2" class="bg-green-500/10 border border-green-500/30 rounded-lg p-6 transition duration-300 hover:scale-105" :class="$clicks < 2 ? 'translate-y-5 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-green-400 font-bold mb-2">✅ 配置驱动</div>
<div class="text-sm opacity-80">
通过 YAML 配置文件定义所有实验
</div>
</div>

<div v-click="3" class="bg-green-500/10 border border-green-500/30 rounded-lg p-6 transition duration-300 hover:scale-105" :class="$clicks < 3 ? 'translate-y-5 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-green-400 font-bold mb-2">✅ 开箱即用</div>
<div class="text-sm opacity-80">
12 个预定义配置模板，快速启动
</div>
</div>

<div v-click="4" class="bg-green-500/10 border border-green-500/30 rounded-lg p-6 transition duration-300 hover:scale-105" :class="$clicks < 4 ? 'translate-y-5 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-green-400 font-bold mb-2">✅ 医学专用</div>
<div class="text-sm opacity-80">
支持多视图、多实例、注意力监督等场景
</div>
</div>

</div>

---
glowSeed: 33
glow: center
glowPreset: blue
---

# 核心设计理念

<div class="grid grid-cols-3 gap-6 mt-12">

<div v-click="1" class="text-center transition duration-500" :class="$clicks < 1 ? 'scale-80 opacity-0' : 'scale-100 opacity-100'">
<div class="text-6xl mb-4">🧩</div>
<div class="text-xl font-bold mb-2">模块化</div>
<div class="text-sm opacity-70">
Backbones、Fusion、Heads 完全独立
</div>
</div>

<div v-click="2" class="text-center transition duration-500" :class="$clicks < 2 ? 'scale-80 opacity-0' : 'scale-100 opacity-100'">
<div class="text-6xl mb-4">⚙️</div>
<div class="text-xl font-bold mb-2">配置驱动</div>
<div class="text-sm opacity-70">
YAML 配置定义一切，无需改代码
</div>
</div>

<div v-click="3" class="text-center transition duration-500" :class="$clicks < 3 ? 'scale-80 opacity-0' : 'scale-100 opacity-100'">
<div class="text-6xl mb-4">🔌</div>
<div class="text-xl font-bold mb-2">可插拔</div>
<div class="text-sm opacity-70">
任意组合组件，快速验证想法
</div>
</div>

</div>

---
glowSeed: 77
glow: left
glowPreset: cyan
---

# 架构组件

<div class="grid grid-cols-4 gap-3 mt-8 text-sm">

<div v-click="1" class="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 transition duration-300 hover:bg-blue-500/20" :class="$clicks < 1 ? 'translate-x--10 opacity-0' : 'translate-x-0 opacity-100'">
<div class="text-blue-400 font-bold mb-2">Backbones</div>
<ul class="space-y-1 opacity-80">
<li>• ResNet (2D/3D)</li>
<li>• EfficientNet</li>
<li>• Vision Transformer</li>
<li>• Swin Transformer</li>
<li>• DenseNet</li>
<li>• MobileNet</li>
</ul>
</div>

<div v-click="2" class="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 transition duration-300 hover:bg-blue-500/20" :class="$clicks < 2 ? 'translate-x--10 opacity-0' : 'translate-x-0 opacity-100'">
<div class="text-blue-400 font-bold mb-2">Fusion</div>
<ul class="space-y-1 opacity-80">
<li>• Concatenate</li>
<li>• Gated Fusion</li>
<li>• Attention Fusion</li>
<li>• Cross-Attention</li>
<li>• Bilinear Pooling</li>
</ul>
</div>

<div v-click="3" class="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 transition duration-300 hover:bg-blue-500/20" :class="$clicks < 3 ? 'translate-x--10 opacity-0' : 'translate-x-0 opacity-100'">
<div class="text-blue-400 font-bold mb-2">Heads</div>
<ul class="space-y-1 opacity-80">
<li>• Classification</li>
<li>• Cox Survival</li>
<li>• Deep Survival</li>
<li>• Discrete Time</li>
</ul>
</div>

<div v-click="4" class="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 transition duration-300 hover:bg-purple-500/20" :class="$clicks < 4 ? 'translate-x--10 opacity-0' : 'translate-x-0 opacity-100'">
<div class="text-purple-400 font-bold mb-2">MIL Aggregators</div>
<ul class="space-y-1 opacity-80">
<li>• Mean Pooling</li>
<li>• Max Pooling</li>
<li>• Attention-based</li>
<li>• Gated Attention</li>
</ul>
</div>

</div>

---
glowSeed: 55
glow: right
glowPreset: blue
---

# 模型构建方式

<div class="grid grid-cols-3 gap-4 mt-8">

<div v-click="1" class="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5 transition duration-300 hover:border-blue-400" :class="$clicks < 1 ? 'translate-y-10 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-blue-400 font-bold mb-3">方式 1: Builder</div>
<div class="text-xs opacity-80 font-mono">
```python
builder = MultiModalModelBuilder(
    num_classes=2
)
builder.add_modality(
    "ct", 
    backbone="swin3d_tiny"
)
builder.set_fusion("attention")
model = builder.build()
```
</div>
<div class="mt-2 text-xs text-green-400">✓ 推荐：灵活且类型安全</div>
</div>

<div v-click="2" class="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5 transition duration-300 hover:border-blue-400" :class="$clicks < 2 ? 'translate-y-10 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-blue-400 font-bold mb-3">方式 2: 配置文件</div>
<div class="text-xs opacity-80 font-mono">
```python
config = load_yaml(
    "configs/smurf.yaml"
)
model = build_model_from_config(
    config
)
```
</div>
<div class="mt-2 text-xs text-green-400">✓ 最佳：可复现性强</div>
</div>

<div v-click="3" class="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5 transition duration-300 hover:border-blue-400" :class="$clicks < 3 ? 'translate-y-10 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-blue-400 font-bold mb-3">方式 3: 直接构造</div>
<div class="text-xs opacity-80 font-mono">
```python
model = GenericMultiModalModel(
    backbones={...},
    fusion=fusion_module,
    head=head_module
)
```
</div>
<div class="mt-2 text-xs text-yellow-400">⚠ 高级：完全控制</div>
</div>

</div>

---
glowSeed: 99
glow: topmost
glowPreset: cyan
---

# 性能指标

<div class="grid grid-cols-3 gap-8 mt-16">

<div v-click="1" class="text-center transition duration-500" :class="$clicks < 1 ? 'scale-50 opacity-0' : 'scale-100 opacity-100'">
<div class="text-5xl font-bold text-yellow-400 mb-2">350+</div>
<div class="text-sm opacity-70">可配置组合</div>
<div class="text-xs opacity-50 mt-1">29 backbones × 5 fusion × 5 aggregators</div>
</div>

<div v-click="2" class="text-center transition duration-500" :class="$clicks < 2 ? 'scale-50 opacity-0' : 'scale-100 opacity-100'">
<div class="text-5xl font-bold text-lime-400 mb-2">3 步</div>
<div class="text-sm opacity-70">快速启动</div>
<div class="text-xs opacity-50 mt-1">安装 → 准备数据 → 开始训练</div>
</div>

<div v-click="3" class="text-center transition duration-500" :class="$clicks < 3 ? 'scale-50 opacity-0' : 'scale-100 opacity-100'">
<div class="text-5xl font-bold text-cyan-400 mb-2">5 种</div>
<div class="text-sm opacity-70">医学场景</div>
<div class="text-xs opacity-50 mt-1">多角度 CT、时间序列、多模态等</div>
</div>

</div>

---
glowSeed: 22
glow: center
glowPreset: blue
---

# 技术栈

<div class="grid grid-cols-4 gap-6 mt-12 text-center">

<div v-click="1" class="transition duration-300" :class="$clicks < 1 ? 'translate-y-10 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-4xl mb-2">🔥</div>
<div class="font-bold">PyTorch 2.0+</div>
<div class="text-xs opacity-60">深度学习框架</div>
</div>

<div v-click="2" class="transition duration-300" :class="$clicks < 2 ? 'translate-y-10 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-4xl mb-2">⚡</div>
<div class="font-bold">FastAPI</div>
<div class="text-xs opacity-60">Web UI 后端</div>
</div>

<div v-click="3" class="transition duration-300" :class="$clicks < 3 ? 'translate-y-10 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-4xl mb-2">⚛️</div>
<div class="font-bold">React 18</div>
<div class="text-xs opacity-60">前端界面</div>
</div>

<div v-click="4" class="transition duration-300" :class="$clicks < 4 ? 'translate-y-10 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-4xl mb-2">🦀</div>
<div class="font-bold">Rust</div>
<div class="text-xs opacity-60">性能加速</div>
</div>

</div>

<div v-click="5" class="mt-12 text-center opacity-70 transition duration-500" :class="$clicks < 5 ? 'opacity-0' : 'opacity-70'">
<div class="text-sm">支持 29+ 视觉骨干网络</div>
<div class="text-xs mt-2">ResNet • EfficientNet • ViT • Swin • DenseNet • MobileNet • ConvNeXt • ...</div>
</div>

---
glowSeed: 11
glow: bottom
glowPreset: cyan
---

# 快速开始

<div class="mt-12 space-y-8">

<div v-click="1" class="flex items-start gap-4 transition duration-300" :class="$clicks < 1 ? 'translate-x--20 opacity-0' : 'translate-x-0 opacity-100'">
<div class="flex-shrink-0 w-12 h-12 rounded-full bg-green-500/20 border border-green-500/50 flex items-center justify-center text-green-400 font-bold">1</div>
<div class="flex-1">
<div class="font-bold mb-1">安装依赖</div>
<div class="text-sm opacity-70 font-mono bg-black/30 px-3 py-2 rounded">uv sync</div>
</div>
</div>

<div v-click="2" class="flex items-start gap-4 transition duration-300" :class="$clicks < 2 ? 'translate-x--20 opacity-0' : 'translate-x-0 opacity-100'">
<div class="flex-shrink-0 w-12 h-12 rounded-full bg-green-500/20 border border-green-500/50 flex items-center justify-center text-green-400 font-bold">2</div>
<div class="flex-1">
<div class="font-bold mb-1">准备数据</div>
<div class="text-sm opacity-70 font-mono bg-black/30 px-3 py-2 rounded">uv run medfusion-preprocess --input-dir data/raw --output-dir data/processed</div>
</div>
</div>

<div v-click="3" class="flex items-start gap-4 transition duration-300" :class="$clicks < 3 ? 'translate-x--20 opacity-0' : 'translate-x-0 opacity-100'">
<div class="flex-shrink-0 w-12 h-12 rounded-full bg-green-500/20 border border-green-500/50 flex items-center justify-center text-green-400 font-bold">3</div>
<div class="flex-1">
<div class="font-bold mb-1">开始训练</div>
<div class="text-sm opacity-70 font-mono bg-black/30 px-3 py-2 rounded">uv run medfusion-train --config configs/default.yaml</div>
</div>
</div>

</div>

---
glowSeed: 66
glow: center
glowPreset: blue
class: text-center
---

# 谢谢观看

<div v-click="1" class="mt-12 space-y-4 transition duration-500" :class="$clicks < 1 ? 'translate-y-10 opacity-0' : 'translate-y-0 opacity-100'">
<div class="text-xl opacity-80">MedFusion - 让医学 AI 研究更简单</div>
<div class="flex justify-center gap-8 mt-8">
<a href="https://github.com/yourusername/medfusion" class="text-blue-400 hover:text-blue-300 transition duration-200">
<div i-carbon:logo-github class="inline text-2xl"/> GitHub
</a>
<a href="https://medfusion.readthedocs.io" class="text-blue-400 hover:text-blue-300 transition duration-200">
<div i-carbon:document class="inline text-2xl"/> 文档
</a>
</div>
</div>
