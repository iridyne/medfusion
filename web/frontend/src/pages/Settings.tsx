import React, { useState, useEffect } from 'react'
import { Card, Radio, Space, Divider, message, Typography, Row, Col, Button, Modal } from 'antd'
import { useTranslation } from 'react-i18next'
import { ThemeMode } from '../theme/config'
import {
  GlobalOutlined,
  BulbOutlined,
  CheckCircleOutlined,
  HistoryOutlined,
} from '@ant-design/icons'
import {
  getUIPreferences,
  resetUIPreferences,
  updateUIPreferences,
  type UIPreferences,
} from '@/api/system'

const { Title, Text } = Typography

interface SettingsProps {
  onThemeChange?: (theme: ThemeMode) => void
}

export default function Settings({ onThemeChange }: SettingsProps) {
  const { t, i18n } = useTranslation()
  const [language, setLanguage] = useState(i18n.language)
  const [themeMode, setThemeMode] = useState<ThemeMode>('auto')
  const [uiPreferences, setUiPreferences] = useState<UIPreferences>({
    history_display_mode: 'friendly',
    language: 'zh',
    theme_mode: 'auto',
  })

  useEffect(() => {
    // 同步语言设置
    setLanguage(i18n.language)
  }, [i18n.language])

  useEffect(() => {
    const load = async () => {
      try {
        const payload = await getUIPreferences()
        setUiPreferences(payload.preferences)
        setLanguage(payload.preferences.language)
        setThemeMode(payload.preferences.theme_mode)
        if (i18n.language !== payload.preferences.language) {
          await i18n.changeLanguage(payload.preferences.language)
        }
      } catch (error) {
        console.error('Failed to load UI preferences:', error)
      }
    }

    void load()
  }, [])

  const handleLanguageChange = (lang: string) => {
    const nextLanguage = lang as UIPreferences['language']
    setLanguage(nextLanguage)
    setUiPreferences((prev) => ({ ...prev, language: nextLanguage }))
    updateUIPreferences({
      ...uiPreferences,
      language: nextLanguage,
    })
      .then(async (payload) => {
        setUiPreferences(payload.preferences)
        await i18n.changeLanguage(payload.preferences.language)
        message.success(t('settings.messages.languageChanged'))
      })
      .catch((error) => {
        console.error('Failed to update language preference:', error)
        message.error(
          language === 'zh' ? '更新语言设置失败' : 'Failed to update language setting',
        )
      })
  }

  const handleThemeChange = (mode: ThemeMode) => {
    setThemeMode(mode)
    setUiPreferences((prev) => ({ ...prev, theme_mode: mode }))
    onThemeChange?.(mode)
    updateUIPreferences({
      ...uiPreferences,
      theme_mode: mode,
    })
      .then((payload) => {
        setUiPreferences(payload.preferences)
        message.success(t('settings.messages.themeChanged'))
      })
      .catch((error) => {
        console.error('Failed to update theme preference:', error)
        message.error(
          language === 'zh' ? '更新主题设置失败' : 'Failed to update theme setting',
        )
      })
  }

  const handleHistoryDisplayModeChange = async (
    mode: UIPreferences['history_display_mode'],
  ) => {
    setUiPreferences((prev) => ({ ...prev, history_display_mode: mode }))
    try {
      const payload = await updateUIPreferences({
        ...uiPreferences,
        history_display_mode: mode,
      })
      setUiPreferences(payload.preferences)
      message.success(
        language === 'zh' ? '版本历史显示方式已更新' : 'History display mode updated',
      )
    } catch (error) {
      console.error('Failed to update UI preferences:', error)
      message.error(
        language === 'zh' ? '更新版本历史显示方式失败' : 'Failed to update history display mode',
      )
    }
  }

  const handleResetPreferences = async () => {
    Modal.confirm({
      title: language === 'zh' ? '恢复默认设置' : 'Reset preferences',
      content:
        language === 'zh'
          ? '这会把语言、主题和版本历史显示方式恢复到默认值，并覆盖当前机器上的设置文件。'
          : 'This will reset language, theme, and version-history display mode to defaults and overwrite the machine-wide settings file.',
      okText: language === 'zh' ? '恢复默认值' : 'Reset',
      cancelText: language === 'zh' ? '取消' : 'Cancel',
      onOk: async () => {
        try {
          const payload = await resetUIPreferences()
          setUiPreferences(payload.preferences)
          setLanguage(payload.preferences.language)
          setThemeMode(payload.preferences.theme_mode)
          await i18n.changeLanguage(payload.preferences.language)
          onThemeChange?.(payload.preferences.theme_mode)
          message.success(
            language === 'zh'
              ? '已恢复默认设置'
              : 'Preferences reset to defaults',
          )
        } catch (error) {
          console.error('Failed to reset UI preferences:', error)
          message.error(
            language === 'zh'
              ? '恢复默认设置失败'
              : 'Failed to reset preferences',
          )
        }
      },
    })
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
        title={
          <Space>
            <HistoryOutlined />
            <span>{language === 'zh' ? '版本历史显示' : 'Version History Display'}</span>
          </Space>
        }
        bordered={false}
        style={{ marginTop: 16 }}
      >
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Text type="secondary">
            {language === 'zh'
              ? '默认隐藏 git / commit 等技术术语，也可以切换成技术视图。当前设置按本机生效，不区分多用户，且目前只作用于自定义模型的版本历史相关页面。'
              : 'Hide git / commit terminology by default, with an option to switch to a technical view. The current preference is machine-wide, not multi-user aware, and only affects custom-model history views for now.'}
          </Text>
          <Radio.Group
            value={uiPreferences.history_display_mode}
            onChange={(e) => handleHistoryDisplayModeChange(e.target.value)}
            size="large"
          >
            <Space direction="vertical" size="middle">
              <Radio value="friendly">
                <Space>
                  <span>{language === 'zh' ? '友好模式' : 'Friendly mode'}</span>
                  {uiPreferences.history_display_mode === 'friendly' && (
                    <CheckCircleOutlined style={{ color: '#52c41a' }} />
                  )}
                </Space>
              </Radio>
              <Radio value="technical">
                <Space>
                  <span>{language === 'zh' ? '技术模式' : 'Technical mode'}</span>
                  {uiPreferences.history_display_mode === 'technical' && (
                    <CheckCircleOutlined style={{ color: '#52c41a' }} />
                  )}
                </Space>
              </Radio>
            </Space>
          </Radio.Group>
        </Space>
      </Card>

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

      <Card
        title={language === 'zh' ? '恢复默认设置' : 'Reset to Defaults'}
        bordered={false}
        style={{ marginTop: 16 }}
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Text type="secondary">
            {language === 'zh'
              ? '如果客户改乱了界面偏好，可以一键恢复默认值。当前会重置语言、主题和版本历史显示方式。'
              : 'If the interface preferences get messy on a client machine, this resets language, theme, and version-history display mode to defaults.'}
          </Text>
          <Button danger onClick={() => void handleResetPreferences()}>
            {language === 'zh' ? '恢复默认设置' : 'Reset to defaults'}
          </Button>
        </Space>
      </Card>

      <div style={{ marginTop: 24, textAlign: 'center' }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {language === 'zh'
            ? '设置会保存到客户主机本地，并按整台机器生效（当前不区分多用户）。'
            : 'Preferences are saved locally on the host machine and currently apply machine-wide (no multi-user split yet).'}
        </Text>
      </div>
    </div>
  )
}
