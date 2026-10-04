// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <span>
#include <string>
#include <vector>
#include <SDL3/SDL.h>

#include "audio_core/common/common.h"
#include "audio_core/sink/sdl3_sink.h"
#include "audio_core/sink/sink_stream.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "core/core.h"

namespace AudioCore::Sink {
/**
 * SDL sink stream, responsible for sinking samples to hardware.
 */
class SDLSinkStream final : public SinkStream {
public:
    /**
     * Create a new sink stream.
     *
     * @param device_channels_ - Number of channels supported by the hardware.
     * @param system_channels_ - Number of channels the audio systems expect.
     * @param output_device    - Name of the output device to use for this stream.
     * @param input_device     - Name of the input device to use for this stream.
     * @param type_            - Type of this stream.
     * @param system_          - Core system.
     * @param event            - Event used only for audio renderer, signalled on buffer consume.
     */
    SDLSinkStream(u32 device_channels_, u32 system_channels_, const std::string& output_device,
                  const std::string& input_device, StreamType type_, Core::System& system_)
        : SinkStream{system_, type_} {
        system_channels = system_channels_;
        device_channels = device_channels_;

        SDL_AudioSpec spec{};
        spec.format = SDL_AUDIO_S16;
        spec.channels = static_cast<int>(device_channels);
        spec.freq = TargetSampleRate;

        std::string device_name{output_device};
        bool capture{false};
        if (type == StreamType::In) {
            device_name = input_device;
            capture = true;
        }

        const SDL_AudioDeviceID device_id = FindDevice(device_name, capture);

        // SDL3 has no per-device buffer size in the spec; request it through a hint instead.
        const std::string sample_frames = std::to_string(TargetSampleCount * 2);
        SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, sample_frames.c_str());

        stream = SDL_OpenAudioDeviceStream(device_id, &spec, &SDLSinkStream::DataCallback, this);

        if (stream == nullptr) {
            LOG_CRITICAL(Audio_Sink, "Error opening SDL audio device: {}", SDL_GetError());
            return;
        }

        SDL_AudioSpec obtained{};
        int obtained_frames{};
        SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(stream), &obtained, &obtained_frames);

        LOG_INFO(Service_Audio,
                 "Opening SDL stream {} with: rate {} channels {} (system channels {}) "
                 " samples {}",
                 SDL_GetAudioStreamDevice(stream), obtained.freq, obtained.channels,
                 system_channels, obtained_frames);
    }

    /**
     * Destroy the sink stream.
     */
    ~SDLSinkStream() override {
        LOG_DEBUG(Service_Audio, "Destructing SDL stream {}", name);
        Finalize();
    }

    /**
     * Finalize the sink stream.
     */
    void Finalize() override {
        if (stream == nullptr) {
            return;
        }

        Stop();
        // Destroying a stream created by SDL_OpenAudioDeviceStream also closes its device.
        SDL_DestroyAudioStream(stream);
        stream = nullptr;
    }

    /**
     * Start the sink stream.
     *
     * @param resume - Set to true if this is resuming the stream a previously-active stream.
     *                 Default false.
     */
    void Start(bool resume = false) override {
        if (stream == nullptr || !paused) {
            return;
        }

        paused = false;
        SDL_ResumeAudioStreamDevice(stream);
    }

    /**
     * Stop the sink stream.
     */
    void Stop() override {
        if (stream == nullptr || paused) {
            return;
        }
        SignalPause();
        SDL_PauseAudioStreamDevice(stream);
    }

private:
    /**
     * Find the SDL device with the given name.
     *
     * @param name    - Device name, empty for the default device.
     * @param capture - True to look for a recording device, false for playback.
     *
     * @return The device id, or the default playback/recording device if the name is not found.
     */
    static SDL_AudioDeviceID FindDevice(const std::string& name, bool capture) {
        const SDL_AudioDeviceID default_device =
            capture ? SDL_AUDIO_DEVICE_DEFAULT_RECORDING : SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
        if (name.empty()) {
            return default_device;
        }

        int count{};
        SDL_AudioDeviceID* devices =
            capture ? SDL_GetAudioRecordingDevices(&count) : SDL_GetAudioPlaybackDevices(&count);
        if (devices == nullptr) {
            return default_device;
        }

        SDL_AudioDeviceID found = default_device;
        for (int i = 0; i < count; ++i) {
            const char* device_name = SDL_GetAudioDeviceName(devices[i]);
            if (device_name != nullptr && name == device_name) {
                found = devices[i];
                break;
            }
        }
        SDL_free(devices);
        return found;
    }

    /**
     * Main callback from SDL. Either needs samples from us (audio render/audio out), or has
     * samples for us to read (audio in).
     *
     * @param userdata          - Custom data pointer passed along, points to a SDLSinkStream.
     * @param stream            - The SDL stream to feed or drain.
     * @param additional_amount - Bytes needed (playback) or bytes available (recording).
     */
    static void DataCallback(void* userdata, SDL_AudioStream* stream, int additional_amount, int /*total_amount*/) {
        auto* impl = static_cast<SDLSinkStream*>(userdata);

        if (!impl || additional_amount <= 0) {
            return;
        }

        const std::size_t num_channels = impl->GetDeviceChannels();
        const std::size_t frame_size = num_channels;
        const std::size_t num_frames{static_cast<std::size_t>(additional_amount) / num_channels /
                                     sizeof(s16)};
        if (num_frames == 0) {
            return;
        }

        impl->buffer.resize(num_frames * frame_size);

        if (impl->type == StreamType::In) {
            const int bytes = static_cast<int>(impl->buffer.size() * sizeof(s16));
            const int read = SDL_GetAudioStreamData(stream, impl->buffer.data(), bytes);
            if (read <= 0) {
                return;
            }
            const std::size_t frames_read =
                static_cast<std::size_t>(read) / num_channels / sizeof(s16);
            std::span<const s16> input_buffer{impl->buffer.data(), frames_read * frame_size};
            impl->ProcessAudioIn(input_buffer, frames_read);
        } else {
            std::span<s16> output_buffer{impl->buffer.data(), num_frames * frame_size};
            impl->ProcessAudioOutAndRender(output_buffer, num_frames);
            SDL_PutAudioStreamData(stream, impl->buffer.data(),
                                   static_cast<int>(output_buffer.size_bytes()));
        }
    }

    /// SDL stream bound to the opened input/output device
    SDL_AudioStream* stream{};

    /// Scratch buffer shared with the SDL audio thread callback
    std::vector<s16> buffer;
};

SDLSink::SDLSink(std::string_view target_device_name) {
    if (!SDL_WasInit(SDL_INIT_AUDIO)) {
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            LOG_CRITICAL(Audio_Sink, "SDL_InitSubSystem audio failed: {}", SDL_GetError());
            return;
        }
    }

    if (target_device_name != auto_device_name && !target_device_name.empty()) {
        output_device = target_device_name;
    } else {
        output_device.clear();
    }

    device_channels = 2;
}

SDLSink::~SDLSink() = default;

SinkStream* SDLSink::AcquireSinkStream(Core::System& system, u32 system_channels_,
                                       const std::string&, StreamType type) {
    system_channels = system_channels_;
    SinkStreamPtr& stream = sink_streams.emplace_back(std::make_unique<SDLSinkStream>(
        device_channels, system_channels, output_device, input_device, type, system));
    return stream.get();
}

void SDLSink::CloseStream(SinkStream* stream) {
    for (size_t i = 0; i < sink_streams.size(); i++) {
        if (sink_streams[i].get() == stream) {
            sink_streams[i].reset();
            sink_streams.erase(sink_streams.begin() + i);
            break;
        }
    }
}

void SDLSink::CloseStreams() {
    sink_streams.clear();
}

f32 SDLSink::GetDeviceVolume() const {
    if (sink_streams.empty() || !sink_streams[0]) {
        return 1.0f;
    }

    return sink_streams[0]->GetDeviceVolume();
}

void SDLSink::SetDeviceVolume(f32 volume) {
    for (auto& stream : sink_streams) {
        stream->SetDeviceVolume(volume);
    }
}

void SDLSink::SetSystemVolume(f32 volume) {
    for (auto& stream : sink_streams) {
        stream->SetSystemVolume(volume);
    }
}

std::vector<std::string> ListSDLSinkDevices(bool capture) {
    std::vector<std::string> device_list;

    if (!SDL_WasInit(SDL_INIT_AUDIO)) {
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            LOG_CRITICAL(Audio_Sink, "SDL_InitSubSystem audio failed: {}", SDL_GetError());
            return {};
        }
    }

    int device_count{};
    SDL_AudioDeviceID* devices = capture ? SDL_GetAudioRecordingDevices(&device_count)
                                         : SDL_GetAudioPlaybackDevices(&device_count);
    if (devices == nullptr) {
        return device_list;
    }
    for (int i = 0; i < device_count; ++i) {
        if (const char* name = SDL_GetAudioDeviceName(devices[i])) {
            device_list.emplace_back(name);
        }
    }
    SDL_free(devices);

    return device_list;
}

bool IsSDLSuitable() {
#if !defined(HAVE_SDL3)
    return false;
#else
    // Check SDL can init
    if (!SDL_WasInit(SDL_INIT_AUDIO)) {
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            LOG_ERROR(Audio_Sink, "SDL failed to init, it is not suitable. Error: {}",
                      SDL_GetError());
            return false;
        }
    }

    // We can set any latency frequency we want with SDL, so no need to check that.

    // Check we can open a device with standard parameters
    SDL_AudioSpec spec{};
    spec.format = SDL_AUDIO_S16;
    spec.channels = 2;
    spec.freq = TargetSampleRate;

    SDL_AudioStream* stream =
        SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);

    if (stream == nullptr) {
        LOG_ERROR(Audio_Sink, "SDL failed to open a device, it is not suitable. Error: {}",
                  SDL_GetError());
        return false;
    }

    SDL_DestroyAudioStream(stream);
    return true;
#endif
}

} // namespace AudioCore::Sink
