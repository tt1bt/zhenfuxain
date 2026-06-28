# thesis_defense_ppt - Design Specification

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | thesis_defense_ppt |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 16 |
| **Design Style** | General Consulting + Academic Minimalist Tech |
| **Target Audience** | 答辩委员会教授、领域专家 |
| **Use Case** | 毕业论文答辩 |
| **Created Date** | 2026-05-17 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 |
| **viewBox** | 0 0 1280 720 |
| **Margins** | left/right 50px, top/bottom 40px |
| **Content Area** | 1180×640 |

---

## III. Visual Theme

### Theme Style

- **Style**: General Consulting + Academic Minimalist Tech
- **Theme**: Light theme
- **Tone**: Professional, academic, tech-innovative

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FFFFFF` | Page background |
| **Secondary bg** | `#F5F7FA` | Card/section background |
| **Primary** | `#1A5F7A` | Title, key sections, icons |
| **Accent** | `#3498DB` | Data highlights, key info |
| **Secondary accent** | `#2ECC71` | Positive indicators |
| **Body text** | `#2C3E50` | Main body text |
| **Secondary text** | `#7F8C8D` | Captions, annotations |
| **Tertiary text** | `#95A5A6` | Footnotes, supplementary |
| **Border/divider** | `#E0E6ED` | Card borders, dividers |
| **Warning** | `#E74C3C` | Issue markers |

### Gradient Scheme

```xml
<linearGradient id="titleGradient" x1="0%" y1="0%" x2="100%" y2="0%">
  <stop offset="0%" stop-color="#1A5F7A"/>
  <stop offset="100%" stop-color="#3498DB"/>
</linearGradient>

<linearGradient id="accentGradient" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#3498DB"/>
  <stop offset="100%" stop-color="#2ECC71"/>
</linearGradient>
```

---

## IV. Typography System

### Font Plan

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Body** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Emphasis** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Code** | — | `Consolas` | `monospace` |

### Font Size Hierarchy

**Baseline**: Body font size = 20px

| Purpose | Ratio to body | Example @ body=20 |
| ------- | ------------- | ----------------- |
| Cover title | 2.5-3x | 50-60px |
| Chapter title | 2-2.5x | 40-50px |
| Page title | 1.5-2x | 30-40px |
| Subtitle | 1.2-1.5x | 24-30px |
| **Body content** | **1x** | **20px** |
| Annotation | 0.7-0.85x | 14-17px |
| Page number | 0.5-0.65x | 10-13px |

---

## V. Layout Principles

### Page Structure

- **Header area**: 80px height - page title and section indicator
- **Content area**: 520px height - main content
- **Footer area**: 40px height - page number and navigation

### Layout Pattern Library

| Pattern | Usage |
| ------- | ----- |
| **Single column centered** | Cover, conclusion, key points |
| **Asymmetric split (3:7)** | Data chart vs. brief takeaway |
| **Three column cards** | Feature lists, parallel points |
| **Matrix grid (2×2)** | Comparisons, classifications |
| **Center-radiating** | Core concept diagrams |

### Spacing Specification

| Element | Value |
| ------- | ----- |
| Safe margin from canvas edge | 50px |
| Content block gap | 24px |
| Icon-text gap | 12px |
| Card gap | 20px |
| Card padding | 24px |
| Card border radius | 12px |

---

## VI. Icon Usage Specification

### Source
- **Built-in icon library**: `templates/icons/`

### Recommended Icon List

| Purpose | Icon Path |
| ------- | --------- |
| Research background | `chunk-filled/search` |
| Literature review | `chunk-filled/book` |
| Methodology | `chunk-filled/settings` |
| Experiment | `chunk-filled/chart` |
| Conclusion | `chunk-filled/flag` |

---

## VII. Visualization Reference List

| Page | Template | Path | Usage |
| ---- | -------- | ---- | ----- |
| P07 | grouped_bar_chart | `templates/charts/grouped_bar_chart.svg` | 实验结果对比 |
| P08 | line_chart | `templates/charts/line_chart.svg` | 性能趋势 |
| P10 | flow_chart | `templates/charts/flow_chart.svg` | 算法流程 |

---

## VIII. Image Resource List

| Filename | Purpose | Type | Acquire Via | Status |
| -------- | ------- | ---- | ----------- | ------ |
| cover_bg.svg | 封面背景 | Decorative | svg-drawn | Existing |

---

## IX. Content Outline

### Slide 01 - 封面

- **Layout**: Full-bleed background with centered title
- **Title**: 面向长尾遥感图像检索的加权哈希学习算法研究
- **Subtitle**: 毕业论文答辩
- **Info**: [姓名] | [学号] | [导师] | [学院]

### Slide 02 - 目录

- **Layout**: Left-aligned numbered list with icons
- **Title**: 目录
- **Content**:
  1. 研究背景与意义
  2. 国内外研究现状
  3. 研究方法与创新点
  4. 实验设计与结果分析
  5. 结论与展望
  6. 致谢

### Slide 03 - 研究背景

- **Layout**: Left text + right visual
- **Title**: 研究背景
- **Content**:
  - 遥感图像数据规模持续增长
  - 类别分布呈现明显长尾特性
  - 灾害监测、异常目标发现等实际应用需求

### Slide 04 - 问题提出

- **Layout**: Cards layout
- **Title**: 问题提出
- **Content**:
  - 深度哈希方法存储和检索效率优势
  - 现有方法在长尾数据上的局限性
  - 尾部类别检索性能下降的挑战

### Slide 05 - 研究意义

- **Layout**: Centered key points
- **Title**: 研究意义
- **Content**:
  - 提升长尾遥感图像检索整体性能
  - 为实际应用提供有效检索方案
  - 推动哈希学习在不平衡数据上的研究

### Slide 06 - 深度哈希学习研究现状

- **Layout**: Three columns
- **Title**: 深度哈希学习研究现状
- **Content**:
  - 监督哈希方法（SH、DSH、CNNH）
  - 无监督哈希方法（ITQ、SDH）
  - 半监督哈希方法

### Slide 07 - 长尾学习研究现状

- **Layout**: Three columns
- **Title**: 长尾学习研究现状
- **Content**:
  - 数据层面：重采样、数据增强
  - 特征层面：迁移学习、元学习
  - 损失层面：类别加权、均衡损失

### Slide 08 - 向心式哈希学习

- **Layout**: Left concept + right diagram
- **Title**: 向心式哈希学习
- **Content**:
  - 核心思想：引入类别中心
  - 优势：训练稳定、结构简洁
  - 不足：未考虑类别样本规模差异

### Slide 09 - 向心式哈希框架

- **Layout**: Architecture diagram
- **Title**: 向心式哈希学习框架
- **Content**:
  - 模型结构：ResNet34 + 哈希层 + 分类器
  - 向心损失：推动样本向类别中心聚集
  - 损失函数：L = L_hash + α × L_cls

### Slide 10 - 加权改进方案

- **Layout**: Formula + explanation
- **Title**: 加权改进方案
- **Content**:
  - 核心思想：基于有效样本数的类别加权
  - 有效样本数公式：E_n = (1 - β^n) / (1 - β)
  - 加权向心损失：不同类别采用不同权重

### Slide 11 - 创新点

- **Layout**: Three cards
- **Title**: 创新点
- **Content**:
  - 将类别均衡思想引入向心式哈希学习
  - 从损失函数层面解决长尾问题
  - 提升尾部类别检索性能

### Slide 12 - 实验设置

- **Layout**: Data table
- **Title**: 实验设置
- **Content**:
  - 数据集：PatternNet、NWPU-RESISC45、RSSCN7、CLRS
  - 评估指标：mAP、P@K、R@K、PR曲线
  - 对比方法：传统哈希方法、向心式哈希

### Slide 13 - 实验结果

- **Layout**: Bar charts + key findings
- **Title**: 实验结果分析
- **Content**:
  - 整体性能对比：加权方法 vs 原始方法
  - 类别段分析：头部、中部、尾部类别性能
  - 参数敏感性分析：β值、α值的影响

### Slide 14 - 消融实验

- **Layout**: Comparison table
- **Title**: 消融实验分析
- **Content**:
  - 不同加权策略的对比
  - 损失函数各组成部分的贡献
  - 不同数据集上的泛化性能

### Slide 15 - 结论与展望

- **Layout**: Two sections
- **Title**: 结论与展望
- **Content**:
  - 研究结论：方法有效、尾部改善、泛化良好
  - 未来展望：注意力机制、跨模态检索、动态加权

### Slide 16 - 致谢

- **Layout**: Centered
- **Title**: 致谢
- **Content**:
  - 感谢导师的悉心指导
  - 感谢实验室同学的帮助
  - 感谢家人的支持

---

## X. Speaker Notes Requirements

- 每页需包含关键要点和过渡语句
- 时间控制提示：每页约1-2分钟
- 重点强调创新点和实验结果

---

## XI. Technical Constraints

1. viewBox: `0 0 1280 720`
2. Background uses `<rect>` elements
3. Text wrapping uses `<tspan>`
4. Transparency uses `fill-opacity`
5. Forbidden: mask, style, class, foreignObject
6. Text: raw Unicode characters
