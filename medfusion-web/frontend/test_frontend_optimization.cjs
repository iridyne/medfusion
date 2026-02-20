#!/usr/bin/env node

/**
 * MedFusion Web UI 前端优化功能测试脚本
 *
 * 测试内容：
 * 1. 文件存在性检查
 * 2. 依赖配置检查
 * 3. 语言包完整性检查
 * 4. 组件导出检查
 * 5. 配置文件验证
 */

const fs = require('fs');
const path = require('path');

// 颜色输出
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function success(message) {
  log(`✅ ${message}`, 'green');
}

function error(message) {
  log(`❌ ${message}`, 'red');
}

function info(message) {
  log(`ℹ️  ${message}`, 'blue');
}

function warning(message) {
  log(`⚠️  ${message}`, 'yellow');
}

// 测试结果统计
let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

function test(name, fn) {
  totalTests++;
  try {
    fn();
    passedTests++;
    success(`${name}`);
    return true;
  } catch (err) {
    failedTests++;
    error(`${name}`);
    console.log(`   ${err.message}`);
    return false;
  }
}

// 辅助函数
function fileExists(filePath) {
  const fullPath = path.join(__dirname, filePath);
  if (!fs.existsSync(fullPath)) {
    throw new Error(`文件不存在: ${filePath}`);
  }
}

function readJSON(filePath) {
  const fullPath = path.join(__dirname, filePath);
  const content = fs.readFileSync(fullPath, 'utf-8');
  return JSON.parse(content);
}

function readFile(filePath) {
  const fullPath = path.join(__dirname, filePath);
  return fs.readFileSync(fullPath, 'utf-8');
}

// 开始测试
log('\n========================================', 'cyan');
log('  MedFusion 前端优化功能测试', 'cyan');
log('========================================\n', 'cyan');

// ==================== 1. 文件存在性检查 ====================
log('\n📁 1. 文件存在性检查\n', 'yellow');

const requiredFiles = [
  'src/components/VirtualList.tsx',
  'src/components/LazyChart.tsx',
  'src/i18n/config.ts',
  'src/i18n/locales/zh.json',
  'src/i18n/locales/en.json',
  'src/theme/config.ts',
  'src/pages/Settings.tsx',
  'package.json',
];

requiredFiles.forEach(file => {
  test(`文件存在: ${file}`, () => fileExists(file));
});

// ==================== 2. 依赖配置检查 ====================
log('\n📦 2. 依赖配置检查\n', 'yellow');

test('package.json 可读取', () => {
  const pkg = readJSON('package.json');
  if (!pkg.dependencies) {
    throw new Error('package.json 缺少 dependencies 字段');
  }
});

const requiredDependencies = [
  'react-window',
  'react-virtualized-auto-sizer',
  'react-i18next',
  'i18next',
];

const requiredDevDependencies = [
  '@types/react-window',
];

test('生产依赖完整性', () => {
  const pkg = readJSON('package.json');
  const missing = requiredDependencies.filter(dep => !pkg.dependencies[dep]);
  if (missing.length > 0) {
    throw new Error(`缺少依赖: ${missing.join(', ')}`);
  }
});

test('开发依赖完整性', () => {
  const pkg = readJSON('package.json');
  const missing = requiredDevDependencies.filter(dep => !pkg.devDependencies[dep]);
  if (missing.length > 0) {
    throw new Error(`缺少开发依赖: ${missing.join(', ')}`);
  }
});

// ==================== 3. 语言包完整性检查 ====================
log('\n🌐 3. 语言包完整性检查\n', 'yellow');

test('中文语言包可读取', () => {
  const zh = readJSON('src/i18n/locales/zh.json');
  if (Object.keys(zh).length === 0) {
    throw new Error('中文语言包为空');
  }
});

test('英文语言包可读取', () => {
  const en = readJSON('src/i18n/locales/en.json');
  if (Object.keys(en).length === 0) {
    throw new Error('英文语言包为空');
  }
});

test('语言包结构一致性', () => {
  const zh = readJSON('src/i18n/locales/zh.json');
  const en = readJSON('src/i18n/locales/en.json');

  const zhKeys = Object.keys(zh);
  const enKeys = Object.keys(en);

  if (zhKeys.length !== enKeys.length) {
    throw new Error(`语言包键数量不一致: zh=${zhKeys.length}, en=${enKeys.length}`);
  }

  const missingInEn = zhKeys.filter(key => !en[key]);
  if (missingInEn.length > 0) {
    throw new Error(`英文语言包缺少键: ${missingInEn.join(', ')}`);
  }
});

test('语言包必需模块存在', () => {
  const zh = readJSON('src/i18n/locales/zh.json');
  const requiredModules = ['common', 'nav', 'workflow', 'training', 'models', 'settings'];

  const missing = requiredModules.filter(module => !zh[module]);
  if (missing.length > 0) {
    throw new Error(`语言包缺少模块: ${missing.join(', ')}`);
  }
});

// ==================== 4. 组件代码检查 ====================
log('\n🧩 4. 组件代码检查\n', 'yellow');

test('VirtualList 组件导出', () => {
  const content = readFile('src/components/VirtualList.tsx');
  if (!content.includes('export default function VirtualList')) {
    throw new Error('VirtualList 组件未正确导出');
  }
  if (!content.includes('react-window')) {
    throw new Error('VirtualList 未导入 react-window');
  }
});

test('LazyChart 组件导出', () => {
  const content = readFile('src/components/LazyChart.tsx');
  if (!content.includes('export default function LazyChart')) {
    throw new Error('LazyChart 组件未正确导出');
  }
  if (!content.includes('IntersectionObserver')) {
    throw new Error('LazyChart 未使用 IntersectionObserver');
  }
});

test('Settings 页面导出', () => {
  const content = readFile('src/pages/Settings.tsx');
  if (!content.includes('export default function Settings')) {
    throw new Error('Settings 页面未正确导出');
  }
  if (!content.includes('useTranslation')) {
    throw new Error('Settings 未使用 useTranslation');
  }
});

// ==================== 5. 配置文件验证 ====================
log('\n⚙️  5. 配置文件验证\n', 'yellow');

test('i18n 配置文件', () => {
  const content = readFile('src/i18n/config.ts');
  if (!content.includes('i18next')) {
    throw new Error('i18n 配置未导入 i18next');
  }
  if (!content.includes('initReactI18next')) {
    throw new Error('i18n 配置未导入 initReactI18next');
  }
  if (!content.includes('localStorage')) {
    throw new Error('i18n 配置未使用 localStorage 持久化');
  }
});

test('主题配置文件', () => {
  const content = readFile('src/theme/config.ts');
  if (!content.includes('lightTheme')) {
    throw new Error('主题配置缺少 lightTheme');
  }
  if (!content.includes('darkTheme')) {
    throw new Error('主题配置缺少 darkTheme');
  }
  if (!content.includes('watchSystemTheme')) {
    throw new Error('主题配置缺少 watchSystemTheme 函数');
  }
  if (!content.includes('matchMedia')) {
    throw new Error('主题配置未使用 matchMedia API');
  }
});

// ==================== 6. 集成检查 ====================
log('\n🔗 6. 集成检查\n', 'yellow');

test('main.tsx 导入 i18n', () => {
  const content = readFile('src/main.tsx');
  if (!content.includes('./i18n/config')) {
    throw new Error('main.tsx 未导入 i18n 配置');
  }
});

test('App.tsx 集成主题', () => {
  const content = readFile('src/App.tsx');
  if (!content.includes('ConfigProvider')) {
    throw new Error('App.tsx 未使用 ConfigProvider');
  }
  if (!content.includes('useTranslation')) {
    throw new Error('App.tsx 未使用 useTranslation');
  }
});

test('Sidebar 添加设置菜单', () => {
  const content = readFile('src/components/Sidebar.tsx');
  if (!content.includes('SettingOutlined')) {
    throw new Error('Sidebar 未导入 SettingOutlined 图标');
  }
  if (!content.includes('/settings')) {
    throw new Error('Sidebar 未添加设置路由');
  }
  if (!content.includes('useTranslation')) {
    throw new Error('Sidebar 未使用国际化');
  }
});

test('ModelLibrary 使用 VirtualList', () => {
  const content = readFile('src/pages/ModelLibrary.tsx');
  if (!content.includes('VirtualList')) {
    throw new Error('ModelLibrary 未导入 VirtualList');
  }
  if (!content.includes('useTranslation')) {
    throw new Error('ModelLibrary 未使用国际化');
  }
});

test('TrainingMonitor 使用 LazyChart', () => {
  const content = readFile('src/pages/TrainingMonitor.tsx');
  if (!content.includes('LazyChart')) {
    throw new Error('TrainingMonitor 未导入 LazyChart');
  }
  if (!content.includes('useTranslation')) {
    throw new Error('TrainingMonitor 未使用国际化');
  }
});

// ==================== 7. 代码质量检查 ====================
log('\n✨ 7. 代码质量检查\n', 'yellow');

test('VirtualList TypeScript 类型', () => {
  const content = readFile('src/components/VirtualList.tsx');
  if (!content.includes('interface') && !content.includes('type')) {
    throw new Error('VirtualList 缺少 TypeScript 类型定义');
  }
});

test('LazyChart TypeScript 类型', () => {
  const content = readFile('src/components/LazyChart.tsx');
  if (!content.includes('interface') && !content.includes('type')) {
    throw new Error('LazyChart 缺少 TypeScript 类型定义');
  }
});

test('Settings TypeScript 类型', () => {
  const content = readFile('src/pages/Settings.tsx');
  if (!content.includes('ThemeMode')) {
    throw new Error('Settings 缺少 ThemeMode 类型定义');
  }
});

// ==================== 测试结果汇总 ====================
log('\n========================================', 'cyan');
log('  测试结果汇总', 'cyan');
log('========================================\n', 'cyan');

info(`总测试数: ${totalTests}`);
success(`通过: ${passedTests}`);
if (failedTests > 0) {
  error(`失败: ${failedTests}`);
}

const successRate = ((passedTests / totalTests) * 100).toFixed(1);
log(`\n成功率: ${successRate}%\n`, successRate === '100.0' ? 'green' : 'yellow');

if (failedTests === 0) {
  log('🎉 所有测试通过！前端优化功能已正确实现。\n', 'green');
  process.exit(0);
} else {
  log('⚠️  部分测试失败，请检查上述错误信息。\n', 'red');
  process.exit(1);
}
