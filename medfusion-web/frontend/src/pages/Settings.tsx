import React, { useState, useEffect } from 'react'
import { Card, Radio, Space, Divider, message, Typography, Row, Col } from 'antd'
import { useTranslation } from 'react-i18next'
import { ThemeMode, loadThemeMode, saveThemeMode } from '../theme/config'
import {
  GlobalOutlined,
  BulbOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons'

const { Title, Text } = Typography

interface SettingsProps {
  onThemeChange?: (theme: ThemeMode) => void
}

export default function Settings({ onThemeChange }: SettingsProps) {
  const { t, i18n } = useTranslation()
  const [language, setLanguage] = useState(i18n.language)
  const [themeMode, setThemeMode] = useState<ThemeMode>(loadThemeMode())

  useEffect(() => {
    // 同步语言设置
    setLanguage(i18n.language)
  }, [i18n.language])

  const handleLanguageChange = (lang: string) => {
    i18n.changeLanguage(lang)
    localStorage.setItem('language', lang)
    setLanguage(lang)
    message.success(t('settings.messages.languageChanged'))
  }

  const handleThemeChange = (mode: ThemeMode) => {
    setThemeMode(mode)
    saveThemeMode(mode)
    onThemeChange?.(mode)
    message.success(t('settings.messages.themeChanged'))
  }

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}>{t('settings.title')}</Title>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <GlobalOutlined />
                <span>{t('settings.language')}</span>
              </Space>
            }
            bordered={false}
          >
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <div>
                <Text type="secondary">
                  选择界面显示语言 / Select interface language
                </Text>
              </div>
              <Radio.Group
                value={language}
                onChange={(e) => handleLanguageChange(e.target.value)}
                size="large"
              >
                <Space direction="vertical" size="middle">
                  <Radio value="zh">
                    <Space>
                      <span style={{ fontSize: 16 }}>🇨🇳</span>
                      <span>{t('settings.languages.zh')}</span>
                      {language === 'zh' && (
                        <CheckCircleOutlined style={{ color: '#52c41a' }} />
                      )}
                    </Space>
                  </Radio>
                  <Radio value="en">
                    <Space>
                      <span style={{ fontSize: 16 }}>🇺🇸</span>
                      <span>{t('settings.languages.en')}</span>
                      {language === 'en' && (
                        <CheckCircleOutlined style={{ color: '#52c41a' }} />
                      )}
                    </Space>
                  </Radio>
                </Space>
              </Radio.Group>
            </Space>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <BulbOutlined />
                <span>{t('settings.theme')}</span>
              </Space>
            }
            bordered={false}
          >
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <div>
                <Text type="secondary">
                  {language === 'zh'
                    ? '选择界面主题外观'
                    : 'Select interface theme'}
                </Text>
              </div>
              <Radio.Group
                value={themeMode}
                onChange={(e) => handleThemeChange(e.target.value)}
                size="large"
              >
                <Space direction="vertical" size="middle">
                  <Radio value="light">
                    <Space>
                      <span style={{ fontSize: 16 }}>☀️</span>
                      <span>{t('settings.themes.light')}</span>
                      {themeMode === 'light' && (
                        <CheckCircleOutlined style={{ color: '#52c41a' }} />
                      )}
                    </Space>
                  </Radio>
                  <Radio value="dark">
                    <Space>
                      <span style={{ fontSize: 16 }}>🌙</span>
                      <span>{t('settings.themes.dark')}</span>
                      {themeMode === 'dark' && (
                        <CheckCircleOutlined style={{ color: '#52c41a' }} />
                      )}
                    </Space>
                  </Radio>
                  <Radio value="auto">
                    <Space>
                      <span style={{ fontSize: 16 }}>🔄</span>
                      <span>{t('settings.themes.auto')}</span>
                      {themeMode === 'auto' && (
                        <CheckCircleOutlined style={{ color: '#52c41a' }} />
                      )}
                    </Space>
                  </Radio>
                </Space>
              </Radio.Group>
              {themeMode === 'auto' && (
                <div
                  style={{
                    marginTop: 16,
                    padding: 12,
                    backgroundColor: '#f0f2f5',
                    borderRadius: 4,
                  }}
                >
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {language === 'zh'
                      ? '💡 自动模式将根据系统设置切换主题'
                      : '💡 Auto mode will switch theme based on system settings'}
                  </Text>
                </div>
              )}
            </Space>
          </Card>
        </Col>
      </Row>

      <Divider />

      <Card
        title={t('settings.general')}
        bordered={false}
        style={{ marginTop: 16 }}
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Row>
            <Col span={12}>
              <Text strong>{language === 'zh' ? '版本' : 'Version'}</Text>
            </Col>
            <Col span={12}>
              <Text>0.1.0</Text>
            </Col>
          </Row>
          <Row>
            <Col span={12}>
              <Text strong>{language === 'zh' ? '构建日期' : 'Build Date'}</Text>
            </Col>
            <Col span={12}>
              <Text>2024-02-20</Text>
            </Col>
          </Row>
          <Row>
            <Col span={12}>
              <Text strong>{language === 'zh' ? '框架' : 'Framework'}</Text>
            </Col>
            <Col span={12}>
              <Text>MedFusion Web UI</Text>
            </Col>
          </Row>
        </Space>
      </Card>

      <div style={{ marginTop: 24, textAlign: 'center' }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {language === 'zh'
            ? '设置将自动保存到本地存储'
            : 'Settings are automatically saved to local storage'}
        </Text>
      </div>
    </div>
  )
}
