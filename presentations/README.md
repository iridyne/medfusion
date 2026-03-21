# MedFusion 框架演示文稿

使用 Slidev 创建的 MedFusion 框架介绍演示文稿。

## 特性

- ✨ Slidev Seriph 主题（模糊几何动态背景）
- 🎨 中文内容，专业设计
- 📊 完整的框架架构说明
- 💻 代码高亮和动画效果
- 🎯 适合技术讲解和项目展示

## 文件说明

- `slides.md` - 主演示文稿
- `promo-mvp-slides.md` - 面向小红书 / B 站的推广版演示稿
- `promo-scripts.md` - 口播脚本与录屏结构建议
- `style.css` - 自定义样式
- `uno.config.ts` - UnoCSS 配置
- `global-bottom.vue` - 全局底部组件

## 安装依赖

```bash
cd presentations
npm install
```

## 运行演示

```bash
# 开发模式（实时预览）
npm run dev

# 推广版演示稿
npm run dev:promo

# 构建静态文件
npm run build

# 构建推广版
npm run build:promo

# 导出 PDF
npm run export

# 导出推广版 PDF / PNG
npm run export:promo
```

## 访问演示

开发模式启动后，访问：http://localhost:3030

## 演示文稿内容

1. **封面** - 框架介绍
2. **核心理念** - 设计哲学（模块化、配置驱动、可插拔）
3. **架构组件** - Backbones、Fusion、Heads、MIL Aggregators
4. **模型构建** - 三种构建方式（Builder、配置文件、直接构造）
5. **数据处理** - Dataset 系统和缓存机制
6. **训练系统** - Trainer 架构和核心功能
7. **Web UI** - 实时监控和模型管理
8. **配置系统** - YAML 配置和验证
9. **性能优化** - 最佳实践和优化策略
10. **快速开始** - 三步启动训练

推广版演示稿更适合：

- 小红书 60 到 90 秒口播
- B 站 3 到 5 分钟讲解
- Web UI + Slidev 混合录屏

## 快捷键

- `Space` / `→` - 下一页
- `←` - 上一页
- `f` - 全屏
- `o` - 演讲者模式
- `d` - 深色模式切换
- `g` - 跳转到指定页

## 自定义

### 修改主题

编辑 `slides.md` 的 frontmatter：

```yaml
---
theme: seriph  # Seriph 主题（模糊几何背景）
# 其他主题选项：default, apple-basic, shibainu
---
```

### 修改背景

```yaml
---
background: https://source.unsplash.com/collection/94734566/1920x1080
# 或使用本地图片
background: ./images/background.jpg
---
```

### 添加自定义样式

创建 `style.css` 文件：

```css
/* 自定义样式 */
.slidev-layout {
  /* 你的样式 */
}
```

## 导出选项

### 导出 PDF

```bash
npm run export

# 指定输出文件名
slidev export slides.md --output medfusion-presentation.pdf

# 导出为深色模式
slidev export slides.md --dark
```

### 导出 PNG

```bash
slidev export slides.md --format png
```

### 导出为单页 HTML

```bash
slidev build slides.md --base /medfusion/
```

## 演讲者模式

按 `o` 键进入演讲者模式，可以看到：
- 当前幻灯片
- 下一张幻灯片预览
- 演讲者备注
- 计时器

## 技巧

### 添加点击动画

使用 `<v-click>` 或 `<v-clicks>`：

```markdown
<v-clicks>

- 第一点
- 第二点
- 第三点

</v-clicks>
```

### 代码高亮

```markdown
\`\`\`python {all|1-3|5-7|all}
# 代码会按行高亮显示
def hello():
    print("Hello")

def world():
    print("World")
\`\`\`
```

### 两栏布局

```markdown
---
layout: two-cols
---

# 左侧内容

::right::

# 右侧内容
```

## 故障排查

### 端口被占用

```bash
# 指定其他端口
slidev medfusion-framework.md --port 3031
```

### 构建失败

```bash
# 清除缓存
rm -rf .slidev node_modules
npm install
```

## 相关资源

- [Slidev 官方文档](https://sli.dev/)
- [Seriph 主题文档](https://github.com/slidevjs/themes/tree/main/packages/theme-seriph)
- [Carbon Icons](https://icones.js.org/collection/carbon)

## 许可证

MIT License

---

**最后更新**: 2025-02-27
