import machine
import network
# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
import uos
import utime
# noinspection PyUnresolvedReferences
import ujson
import esp


ADC_MODE_VCC = 255
ADC_MODE_ADC = 33


# noinspection PyUnresolvedReferences
def adc_mode(mode=None):
    import flashbdev

    sector_size = flashbdev.bdev.SEC_SIZE
    flash_size = esp.flash_size()  # device dependent
    init_sector = int(flash_size / sector_size - 4)
    data = bytearray(esp.flash_read(init_sector * sector_size, sector_size))
    current_mode = data[107]
    if mode is None:  # Read adc mode
        return current_mode
    if current_mode == mode:
        print("ADC mode already {}.".format(mode))
    else:
        data[107] = mode  # re-write flash
        esp.flash_erase(init_sector)
        esp.flash_write(init_sector * sector_size, data)
        print("ADC mode is now {}. Reset to apply.".format(mode))
    return mode


def format_datetime(datetime_tuple):
    return '{0:04d}-{1:02d}-{2:02d}T{4:02d}:{5:02d}:{6:02d}.{7:03d}Z'.format(*datetime_tuple)


class SyncError(Exception):
    ...


class AppBase:
    DEVICE_TYPE = ...

    CONFIG_FILENAME = 'config.json'
    STATE_FILENAME = 'state'
    LOG_FILENAME = 'history.log'
    DEVICEID_FILENAME = 'device_id'

    WLAN_TIMEOUT = 10
    ADC_MODE = None

    @classmethod
    def state_factory(cls):
        return {
            'last_run_ts': 0,
            'last_sync_attempt_ts': 0,
        }

    @staticmethod
    def initial_setup():
        # noinspection PyUnresolvedReferences
        import ubinascii

        print('INITIAL SETUP')
        ssid = input('WIFI SSID: ')
        passwd = input('WIFI PASS: ')
        with open('credentials.py', 'w') as fp:
            # fp.write('"""WLAN credentials secrets, generated by setup"""\n\n'.format(ssid))
            fp.write('WLAN_SSID: str = \'{}\'\n'.format(ssid))
            fp.write('WLAN_PASS: str = \'{}\'\n'.format(passwd))
            fp.flush()
        print('Credentials saved\n')

        device_id = input('Device ID (empty for random): ')
        if not device_id:
            device_id = ubinascii.hexlify(uos.urandom(8))
        with open('device_id', 'w') as fp:
            fp.write(device_id)
            fp.flush()
        print('New device_id={} saved\n'.format(device_id))

        machine.reset()

    @classmethod
    def ensure_setup(cls):
        try:
            import credentials
        except ImportError:
            has_credentials = False
            print('ERROR: WLAN credentials not found!')
        else:
            has_credentials = True

        try:
            uos.stat(cls.DEVICEID_FILENAME)
        except OSError:
            has_deviceid = False
            print('ERROR: Device ID not found!')
        else:
            has_deviceid = True

        if not has_credentials or not has_deviceid:
            cls.initial_setup()

    @classmethod
    def load_device_id(cls):
        with open(cls.DEVICEID_FILENAME, 'r') as fp:
            device_id = fp.read()
        return device_id

    @classmethod
    def run(cls, *args, **kwargs):
        # noinspection PyArgumentList
        app = cls(*args, **kwargs)
        try:
            app.runtime_main()
        except Exception as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            print('CRITICAL ERROR:\n{}: {}'.format(repr(exc), exc))
            app.on_critical(exc)
            if app.wlan and app.wlan.isconnected():
                app.start_webrepl(force=True)
                app.webrepl_wait(force=True)
            else:
                print('Rebooting in 30s')
                utime.sleep(30)
            machine.reset()

    def on_critical(self, exc):
        pass

    def __init__(self):
        self.ensure_setup()
        import credentials
        self.credentials = credentials
        self.requests = None
        self.device_id = self.load_device_id()
        self.config = {}
        self.rtc = machine.RTC()
        self.wlan = network.WLAN(network.STA_IF)
        self.state = self.state_factory()
        self.lowpower_phase = False
        self.sync_success = None

    # noinspection PyUnresolvedReferences
    def rtc_is_sane(self):
        return self.rtc.datetime()[0] >= 2020

    def load_config(self):
        with open(self.CONFIG_FILENAME, 'r') as fp:
            self.config = ujson.load(fp)

    def save_state(self):
        with open(self.STATE_FILENAME, 'w') as fp:
            ujson.dump(self.state, fp)
            fp.flush()

    def load_state(self, restore=False):
        try:
            with open(self.STATE_FILENAME, 'r') as fp:
                saved_state = ujson.load(fp)
                # Update default state using saved values, discarding saved excess (previous version?) keys
                self.state.update((k, saved_state[k]) for k in self.state if k in saved_state)
        except (OSError, ValueError):
            pass

        if restore:
            self.restore_state()

    def restore_state(self):
        pass

    def connect_wlan(self):
        if not self.wlan.isconnected():  # TODO: use fixed address, disable DHCP overhead time
            self.wlan.active(True)
            ifcfg = self.config['ifconfig']
            self.wlan.ifconfig((ifcfg['address'], ifcfg['mask'], ifcfg['gateway'], ifcfg['dns']))
            print('Connecting WiFi to {}'.format(self.credentials.WLAN_SSID))
            self.wlan.connect(self.credentials.WLAN_SSID, self.credentials.WLAN_PASS)
            for _ in range(int(self.WLAN_TIMEOUT)):
                utime.sleep_ms(1000)
                if self.wlan.isconnected():
                    print('WiFi connected:', self.wlan.ifconfig())
                    break
            else:
                print('WiFi did not connect in {}s'.format(self.WLAN_TIMEOUT))

    @property
    def webrepl_cfg(self):
        return self.config['webrepl']

    def start_webrepl(self, force=False):
        if self.wlan.isconnected():
            if force or self.webrepl_cfg['enable']:
                # noinspection PyUnresolvedReferences
                esp.sleep_type(esp.SLEEP_NONE)
                import webrepl
                # noinspection PyUnresolvedReferences
                webrepl.start(password=self.webrepl_cfg['password'])

    def webrepl_wait(self, force=False):
        if self.wlan.isconnected():
            if force or self.webrepl_cfg['enable']:
                webrepl_delay = self.webrepl_cfg['time_window']
                print('Waiting {}s for a webrepl connection'.format(webrepl_delay))
                utime.sleep(webrepl_delay)

    @staticmethod
    def _fmt_exc(exc):
        return '{}, {}'.format(repr(exc), exc)

    def get_requests(self):
        if self.requests is None:
            import requests
            self.requests = requests
        return self.requests

    def _sync_wrapper(self, sync_fn, fail_msg, import_requests=True):
        requests = self.get_requests() if import_requests else None
        try:
            if self.wlan.isconnected():
                sync_fn(requests)
            else:
                raise SyncError('WLAN not connected')

        except SyncError as exc:
            print('{}: {}'.format(fail_msg, exc))
            self.sync_success = False

        else:
            if self.sync_success is None:
                self.sync_success = True

    def sync_ntp(self):
        def sync_ntp_fn(_):
            # noinspection PyUnresolvedReferences
            import ntptime

            last_exc = None
            for _ in range(3):
                try:
                    ntptime.settime()
                except Exception as exc:
                    last_exc = exc
                else:
                    print('Synchronized RTC to NTP')
                    break
            else:
                raise SyncError(self._fmt_exc(last_exc))

        self._sync_wrapper(sync_ntp_fn, fail_msg='Failed syncing RTC', import_requests=False)

    def base_url(self):
        server_cfg = self.config['server']
        return 'http://{}:{}/{}'.format(server_cfg['address'], server_cfg['port'], self.DEVICE_TYPE)

    def sync_config(self):
        def sync_config_fn(requests):
            nonlocal self

            cfg_mtime = uos.stat(self.CONFIG_FILENAME)[7]
            try:
                cfg_sync_endpoint = '/sync_config'
                print('Checking remote config... ', end='')
                url = self.base_url() + cfg_sync_endpoint + '?only_timestamp'
                response = requests.get(url)
                json_data = response.json()
                remote_mtime = int(json_data['mtime'])
                if cfg_mtime >= remote_mtime:
                    print('ok')
                    return  # last cfg update is newer than latest remote version, skip

                print('syncing')

                url = self.base_url() + cfg_sync_endpoint
                response = requests.get(url)
                json_data = response.json()
                self.config = json_data['config']
                with open(self.CONFIG_FILENAME, 'w') as fp:
                    ujson.dump(self.config, fp)
                    fp.flush()
                print('Config synced, rebooting')
                machine.reset()

            except (OSError, ValueError) as exc:
                raise SyncError(self._fmt_exc(exc))

        self._sync_wrapper(sync_config_fn, fail_msg='Failed syncing config')

    def sync_state(self):
        def sync_state_fn(requests):
            nonlocal self

            try:
                print('Checking remote state... ', end='')
                url = self.base_url() + '/state_set?deviceid=' + self.device_id
                response = requests.get(url)
                json_data = response.json()
                if json_data:
                    # Update current state with remote values, ignoring excess (unknown) keys
                    self.state.update((k, json_data[k]) for k in self.state if k in json_data)
                    # excess_keys = set(json_data.keys()) - set(self.state.keys())
                    # if excess_keys:
                    #     print('WARNING: Remote state has unknown keys: {}'.format(', '.join(excess_keys)))
                    print('synced')
                else:
                    print('empty')

            except (OSError, ValueError) as exc:
                raise SyncError(self._fmt_exc(exc))

        self._sync_wrapper(sync_state_fn, fail_msg='Failed syncing state')

    def save_log(self):
        # noinspection PyTypeChecker
        log_str = self.format_log()  # type: str

        if log_str is None:
            print('No log to save')
            return

        print('Logging:', log_str)
        with open(self.LOG_FILENAME, 'a') as fp:
            fp.write(log_str)
            fp.write('\n')

    def format_log(self):
        return None

    def upload_log(self):
        def upload_log_fn(requests):
            nonlocal self

            print('Uploading log... ', end='')
            log_size = uos.stat(self.LOG_FILENAME)[6]
            url = self.base_url() + '/upload_log?deviceid=' + self.device_id
            response = None
            with open(self.LOG_FILENAME, 'rb') as fp:
                try:
                    response = requests.post(url, stream=fp, bufsize=32,
                                             headers={'Content-Type': 'text/plain', 'Content-Length': log_size})
                except (OSError, ValueError) as exc:
                    raise SyncError(self._fmt_exc(exc))
            if response is not None:
                resp_text = response.text
                try:
                    recvd_bytes = int(resp_text.strip())
                except ValueError:
                    raise SyncError('Bad response')
                else:
                    if recvd_bytes != log_size:
                        raise SyncError('Size mismatch: got {} vs {}'.format(recvd_bytes, log_size))
                    else:
                        with open(self.LOG_FILENAME, 'w'):
                            pass  # blank out log file
                        print('success')

        self._sync_wrapper(upload_log_fn, fail_msg='\nFailed uploading log')
        self.state['last_sync_attempt_ts'] = utime.time()

    def seconds_to_next_sync(self):
        if not self.rtc_is_sane():
            return -1
        return self.config['sync_interval'] - abs(utime.time() - self.state['last_sync_attempt_ts'])

    def ensure_adc_mode(self):
        if self.ADC_MODE is None:
            return
        if adc_mode() != self.ADC_MODE:
            adc_mode(self.ADC_MODE)
            machine.reset()

    # noinspection PyUnresolvedReferences
    def runtime_setup(self):
        self.ensure_adc_mode()

        self.lowpower_phase = machine.reset_cause() in (machine.DEEPSLEEP_RESET, 2)
        print('Reset cause', machine.reset_cause())
        print('Booted in {} phase'.format('lowpower' if self.lowpower_phase else 'normal'))

        if not self.lowpower_phase:
            self.get_requests()

        self.load_config()
        self.load_state(restore=True)

        esp.sleep_type(esp.SLEEP_LIGHT)
        network.WLAN(network.AP_IF).active(False)  # Disable AP

        self.setup()

    def setup(self):
        raise NotImplementedError

    def runtime_main(self):
        self.runtime_setup()

        if self.check_low_bat():
            self.on_low_bat()

        if self.lowpower_phase and self.seconds_to_next_sync() <= 0:
            print('Resetting to sync')
            machine.reset()

        if self.lowpower_phase:
            self.wlan.active(False)
        else:
            self.connect_wlan()
            self.start_webrepl()
            self.sync_ntp()
            self.sync_config()
            self.sync_state()

        self.main()

        self.save_state()
        self.save_log()

        reboot_mode = 4  # low power mode

        if not self.lowpower_phase:
            self.upload_log()
            self.webrepl_wait()
            self.wlan.active(False)

        else:
            sync_time = self.seconds_to_next_sync()
            if sync_time <= 0:
                # print('Sync interval exceeded, syncing after deep sleep')
                reboot_mode = 0
            # else:
            #     print('Next sync in {}s'.format(int(sync_time)))

        self.state['last_run_ts'] = utime.time()

        self.save_state()

        self.before_deepsleep()

        deep_sleep_delay = self.config['update_interval']
        print('uptime {:.2f}s'.format(utime.ticks_ms() / 1000.0))
        print('Rebooting to {} mode in {}s'.format('lowpower' if reboot_mode == 4 else 'normal', deep_sleep_delay))
        # noinspection PyArgumentList
        esp.deepsleep(int(deep_sleep_delay * 1_000_000), reboot_mode)  # Restart in low power mode
        utime.sleep(int(deep_sleep_delay))  # sometimes deepsleep doesn't trigger immediately so we wait too to make sure

    def main(self):
        raise NotImplementedError

    def before_deepsleep(self):
        pass

    def on_low_bat(self):
        pass

    def state_recent_avg(self, key, avg_size, new_val=None):
        # noinspection PyTypeChecker
        values = self.state[key]  # type: list
        if new_val is not None:
            values.append(new_val)
        values = values[-avg_size:]
        # noinspection PyTypeChecker
        self.state[key] = values
        return sum(values) / len(values)

    @property
    def bat_cfg(self):
        return self.config['battery']

    def vcc_read(self):
        vcc_adc = machine.ADC(1)
        vcc_raw = vcc_adc.read()
        if vcc_raw == 65535:
            return None
        return (vcc_raw / 1000) * self.bat_cfg['vcc_adc_cal']

    def battery_level(self):
        vcc = self.vcc_read()
        if vcc is None:
            return None
        vmin = self.bat_cfg['vcc_min']
        vmax = self.bat_cfg['vcc_max']
        bat_nl = max(0.0, min(1.0, (vcc - vmin) / (vmax - vmin)))
        if 0.0 < bat_nl < 1.0:
            # bat = 1.0 - (20 * (1.0 - bat_nl ** 0.025))
            # bat = 1.0 - 5.0 * (1.0 / (bat_nl ** 0.1) - 1.0)
            bat = 1.0555 * 2.7 ** (-6.0 * 0.009 ** bat_nl)  # ~ li-ion 4.25-3.50 range
            bat = max(0.0, min(1.0, bat))
        else:
            bat = bat_nl
        print('bat level = {}'.format(bat))
        return bat

    def check_low_bat(self):
        vcc = self.vcc_read()
        if vcc is None:
            return None
        return vcc < self.bat_cfg['vcc_verylow']

    def get_sched_temp_for_utc(self, datetime_tuple, config_keys):
        hour = (datetime_tuple[4] + self.config['utc_delta']) % 24
        v = self.config
        for k in config_keys:
            v = v.get(k)
        return v[hour]
