// SPDX-FileCopyrightText: Copyright 2016 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <vector>
#include <QColor>
#include <QString>
#include <QWidget>
#include <qobjectdefs.h>
#include "citron/configuration/configuration_shared.h"
#include "common/common_types.h"
#include "common/settings_enums.h"
#include "configuration/shared_translation.h"
#include "video_core/host_device_info.h"

class QPushButton;
class QEvent;
class QObject;
class QComboBox;

namespace Settings {
enum class NvdecEmulation : u32;
enum class RendererBackend : u32;
} // namespace Settings

namespace Core {
class System;
}

namespace Ui {
class ConfigureGraphics;
}

namespace ConfigurationShared {
class Builder;
}

class ConfigureGraphics : public ConfigurationShared::Tab {
    Q_OBJECT

public:
    explicit ConfigureGraphics(
        const Core::System& system_, std::vector<VideoCore::HostDeviceRecord>& records_,
        const std::function<void()>& expose_compute_option_,
        const std::function<void(Settings::AspectRatio, Settings::ResolutionSetup)>&
            update_aspect_ratio_,
        std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group_,
        const ConfigurationShared::Builder& builder, QWidget* parent = nullptr);
    ~ConfigureGraphics() override;

    void ApplyConfiguration() override;
    void SetConfiguration() override;

private:
    void changeEvent(QEvent* event) override;
    void RetranslateUI();

    void Setup(const ConfigurationShared::Builder& builder);

    void PopulateVSyncModeSelection(bool use_setting);
    void UpdateVsyncSetting() const;
    void UpdateBackgroundColorButton(QColor color);
    void UpdateAPILayout();
    void UpdateDeviceSelection(int device);
    void UpdateShaderBackendSelection(int backend);

    void RetrieveVulkanDevices();

    /// Turns a VSync mode into a textual string for the UI
    const QString TranslateVSyncMode(Settings::VSyncMode mode,
                                     Settings::RendererBackend backend) const;

    Settings::RendererBackend GetCurrentGraphicsBackend() const;

    int FindIndex(u32 enumeration, int value) const;

    std::unique_ptr<Ui::ConfigureGraphics> ui;
    QColor bg_color;

    std::vector<std::function<void(bool)>> apply_funcs{};

    std::vector<VideoCore::HostDeviceRecord>& records;
    std::vector<QString> vulkan_devices;
    std::vector<std::vector<Settings::VSyncMode>> device_present_modes;
    std::vector<Settings::VSyncMode>
        vsync_mode_combobox_enum_map{}; //< Keeps track of which VSync mode corresponds to which
                                        // selection in the combobox
    u32 vulkan_device{};
    const std::function<void()>& expose_compute_option;
    const std::function<void(Settings::AspectRatio, Settings::ResolutionSetup)> update_aspect_ratio;

    const Core::System& system;
    const ConfigurationShared::ComboboxTranslationMap& combobox_translations;

    QPushButton* api_restore_global_button;
    QComboBox* vulkan_device_combobox;
    QComboBox* api_combobox;
    QComboBox* vsync_mode_combobox;
    QPushButton* vsync_restore_global_button;
    QWidget* vulkan_device_widget;
    QWidget* api_widget;
    QComboBox* aspect_ratio_combobox;
    QComboBox* resolution_combobox;
    QWidget* fsr_sharpness_widget;
    QWidget* cas_sharpness_widget;
    QWidget* lq_widget;
    std::vector<QWidget*> crt_widgets;

    // This variable will hold the raw stylesheet string
    QString m_template_style_sheet;
};
