import utime
import machine

from src.app_base import AppBase, format_datetime
from src.ntc import NTC


class ThermApp(AppBase):
    DEVICE_TYPE = 'therm'

    @classmethod
    def state_factory(cls):
        state = super().state_factory()
        state.update({
            'operating_mode': 'schedule',  # see therm() for implemented modes
            'therm_switch': False,
            'last_trigger_ts': 0,
            'recent_temps': [],
            'last_tamper_ts': 0,
            'grace_overrides_manual': False,
            'grace_given_hourly': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        return state

    # noinspection PyTypeChecker
    def __init__(self):
        super().__init__()
        self.therm_pin_on = None  # type: machine.Pin
        self.therm_pin_off = None  # type: machine.Pin
        self.ntc = None  # type: NTC
        self.current_temp = None
        self.avg_temp = None
        self.set_temp = None
        self.tamper_detected = False

    def restore_state(self):
        pass
        # self.set_therm_switch(self.state['therm_switch'])

    def format_log(self):
        log_items = [
            format_datetime(self.rtc.datetime()) if self.rtc_is_sane() else '',  # rtc time
            self.state['operating_mode'], # operating mode
            '{:.2f}'.format(self.current_temp),  # current temp
            '{:.2f}'.format(self.avg_temp),  # avg temp
            '{:.2f}'.format(self.set_temp),  # set temp
            'on' if self.state['therm_switch'] else 'off',  # therm switch
            'wlan' if self.wlan.isconnected() else 'nowlan',  # wifi connected
            'tampered' if self.tamper_detected else 'intact',  # tamper detected
            'gracious' if self.should_give_grace() else ('ignored' if self.tamper_detected else 'normal'),  # grace mode
        ]
        return ' '.join(log_items)

    def setup_temp(self, sensor_cfg):
        self.ntc = NTC(pin=sensor_cfg['adc_pin'],
                       ntc_nom_r=sensor_cfg['ntc_nom_r'], ntc_nom_t=sensor_cfg['ntc_nom_t'],
                       ntc_bcoeff=sensor_cfg['ntc_bcoeff'], vdiv_r2=sensor_cfg['ntc_div_r2'],
                       t_offset=sensor_cfg['ntc_offset'], samples=5, samples_delay_us=500)

    def setup_therm_tristate(self):
        self.therm_pin_on = machine.Pin(self.config['thermostat']['on_pin'], mode=machine.Pin.IN)
        self.therm_pin_off = machine.Pin(self.config['thermostat']['off_pin'], mode=machine.Pin.IN)

    def is_tampered(self):
        sense_pin = machine.Pin(self.config['antitamper']['latch_sense_pin'], mode=machine.Pin.IN)
        # noinspection PyArgumentList
        return bool(sense_pin.value())

    def reset_tampered(self):
        print('Resetting antitamper latch')
        reset_pin = machine.Pin(self.config['antitamper']['latch_reset_pin'], mode=machine.Pin.OUT)
        reset_pin.on()
        utime.sleep_us(250)
        reset_pin.off()

    def should_give_grace(self, record=False):
        if not self.rtc_is_sane():
            print('Skipping grace calc as RTC is not sane')
            return False

        hour = (self.rtc.datetime()[4] + self.config['utc_delta']) % 24
        grace_time_left = self.config['antitamper']['grace_delay'] - (utime.time() - self.state['last_tamper_ts'])
        within_grace_period = grace_time_left > 0
        grace_given_hour = self.state['grace_given_hourly'][hour]
        max_hourly_exceeded = grace_given_hour > self.config['antitamper']['grace_delay']
        grace_given_day = sum(self.state['grace_given_hourly'])
        max_daily_exceeded = grace_given_day > self.config['antitamper']['grace_max_daily']

        giving_grace = within_grace_period and not max_hourly_exceeded and not max_daily_exceeded

        if record:
            if giving_grace:
                print('Giving grace for {}s, given {:.0f}s for hour {:02d}:##, {:.0f}s within the last 24h'
                      .format(grace_time_left, grace_given_hour, hour, grace_given_day))
                self.state['grace_given_hourly'][hour] += max(0.0, float(utime.time() - self.state['last_run_ts']))
            elif within_grace_period:
                print('Not giving grace despite within grace period because {} exceeded'.format(
                    ' and '.join('{} limit'.format(d) for l, d
                                 in ((max_hourly_exceeded, 'hourly'), (max_daily_exceeded, 'daily')) if l)
                ))
            # reset next hour (we keep 23 hours)
            next_hour = (hour + 1) % 24
            self.state['grace_given_hourly'][next_hour] = 0.0

        return giving_grace

    def set_therm_switch(self, switched, memoized=False):
        if memoized:
            if self.state['therm_switch'] == switched:
                return
        print('Setting therm switch {}'.format('on' if switched else 'off'))
        impulse_pin = self.therm_pin_on if switched else self.therm_pin_off
        # noinspection PyArgumentList
        impulse_pin.init(mode=machine.Pin.OUT)
        impulse_pin.on()
        utime.sleep_ms(self.config['thermostat']['impulse_ms'])
        impulse_pin.off()
        self.state['therm_switch'] = switched
        # noinspection PyArgumentList
        impulse_pin.init(mode=machine.Pin.IN)
        if self.config['antitamper']['reset_after_therm']:
            utime.sleep_ms(50)
            self.reset_tampered()

    def therm(self, set_temp):
        current_temp = self.ntc.read()
        temps_averaging = self.config['thermostat'].get('temps_average')
        if temps_averaging and isinstance(temps_averaging, int) and temps_averaging > 0:
            avg_temp = self.state_recent_avg('recent_temps', temps_averaging, new_val=current_temp)
        else:
            avg_temp = current_temp

        onoff_fmt = lambda s: 'on' if s else 'off'

        switch_prev_state = self.state['therm_switch']
        grace_mode = self.should_give_grace(record=True)
        grace_overrides_manual = self.state['grace_overrides_manual']

        operating_mode = self.state['operating_mode']
        # TODO: Track last remote command timestamp (send from server) to failsafe to schedule
        if operating_mode == 'remote' and (not grace_overrides_manual or not grace_mode):
            switched = switch_prev_state
            print('Operating in remote mode, switch stays {} as per last received state'.format(onoff_fmt(switched)))
        # elif operating_mode == 'follow':  # TODO: Implement + hardware
        elif grace_mode:
            switched = self.config['antitamper']['grace_switch']
            print('Setting thermostat switch {} as giving grace'.format(onoff_fmt(switched)))

        else:  # default mode, == 'schedule'
            temp_triggered = avg_temp < set_temp
            switch_would_change = switch_prev_state != temp_triggered
            trigger_minseconds = self.config['thermostat']['trigger_minseconds']
            minseconds_exceeded = abs(utime.time() - self.state['last_trigger_ts']) > trigger_minseconds
            switched = temp_triggered if minseconds_exceeded else switch_prev_state
            if switch_would_change:
                if minseconds_exceeded:
                    self.state['last_trigger_ts'] = utime.time()
                else:
                    print('Current temp {:.2f} would have switched {} from {} but threshold of {:.2f} not met'.format(
                        avg_temp, onoff_fmt(temp_triggered), onoff_fmt(switch_prev_state), trigger_minseconds))

        self.set_therm_switch(switched)

        return current_temp, avg_temp, set_temp

    def setup(self):
        self.setup_therm_tristate()
        self.setup_temp(self.config['temp_sensor'])

        # Reset antitamper (ignore) if booting from poweroff
        if machine.reset_cause() in (0, 1) and self.is_tampered():
            self.reset_tampered()

    def main(self):
        if self.is_tampered():
            self.tamper_detected = True
            self.state['last_tamper_ts'] = utime.time()
            print('Tamper detected!')
            self.reset_tampered()

        if self.rtc_is_sane():
            set_temp = self.get_sched_temp_for_utc(self.rtc.datetime(), ('thermostat', 'sched_temps'))
        else:
            sched_temps = self.config['thermostat']['sched_temps']
            set_temp = sum(sched_temps) / len(sched_temps)  # use avg temp

        self.current_temp, self.avg_temp, self.set_temp = self.therm(set_temp)  # thermostat
