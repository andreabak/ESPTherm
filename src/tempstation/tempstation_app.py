import machine
import math
# import utime
import esp
# noinspection PyUnresolvedReferences
import ssd1306
# noinspection PyUnresolvedReferences
import dht

# from ntc import NTC

from src.app_base import AppBase, format_datetime, ADC_MODE_VCC


# noinspection PyPep8Naming
class SSD1306_SPI_custom(ssd1306.SSD1306_SPI):
    def __init__(self, width, height, spi, dc, res, cs, external_vcc=False, resume_mode=False):
        self.rate = 6 * 1024 * 1024
        dc.init(dc.OUT, value=0)
        res.init(res.OUT, value=0)
        cs.init(cs.OUT, value=1)
        self.spi = spi
        self.dc = dc
        self.res = res
        self.cs = cs

        self.resume_mode = resume_mode
        if not self.resume_mode:
            import utime
            self.res(1)
            utime.sleep_ms(5)
            self.res(0)
            utime.sleep_ms(45)
            self.res(1)
            utime.sleep_ms(50)

        ssd1306.SSD1306.__init__(self, width, height, external_vcc)

    def init_display(self):
        if not self.resume_mode:
            super().init_display()


class TempStationApp(AppBase):
    DEVICE_TYPE = 'tempstation'
    ADC_MODE = ADC_MODE_VCC

    @classmethod
    def state_factory(cls):
        state = super().state_factory()
        state.update({
            'recent_temps': [],
            'recent_humidities': []
        })
        return state

    def on_critical(self, exc):
        if self.oled is not None:
            self.oled.fill(0)
            self.oled.text('CRITICAL ERROR:', 0, 0)
            line_chars = self.oled.width // 8
            error_msg = self._fmt_exc(exc)
            lineno = 1
            while error_msg:
                msg = error_msg[:line_chars]
                self.oled.text(msg, 0, lineno * 10)
                error_msg = error_msg[line_chars:]
                lineno += 1
            self.oled.show()

    def format_log(self):
        temp_set = self.get_sched_temp_for_utc(self.rtc.datetime(), ('station', 'sched_temps'))
        vcc = self.vcc_read()
        n_fmt = '{:.2f}'
        log_items = [
            format_datetime(self.rtc.datetime()) if self.rtc_is_sane() else '',
            n_fmt.format(self.current_temp),
            n_fmt.format(self.avg_temp),
            n_fmt.format(self.current_humidity),
            n_fmt.format(self.avg_humidity),
            n_fmt.format(temp_set),
            'wlan' if self.wlan.isconnected() else 'nowlan',
            n_fmt.format(vcc) if vcc else ''
        ]
        return ' '.join(log_items)

    def __init__(self):
        super().__init__()
        self.dht = None
        # self.adc_cal = None
        # self.ntc = None
        self.oled = None

        self.current_temp = None
        self.avg_temp = None
        self.current_humidity = None
        self.avg_humidity = None

    def setup(self):
        dp = machine.Pin(self.config['dht']['pin'])
        self.dht = dht.DHT22(dp)
        # ntc_cfg = self.config['temp_sensor']
        # self.adc_cal = machine.Pin(ntc_cfg['adc_cal_pin'], machine.Pin.OUT, value=0)
        # self.ntc = NTC(pin=ntc_cfg['adc_pin'],
        #                ntc_nom_r=ntc_cfg['ntc_nom_r'],
        #                ntc_nom_t=ntc_cfg['ntc_nom_t'],
        #                ntc_bcoeff=ntc_cfg['ntc_bcoeff'],
        #                vdiv_r2=ntc_cfg['ntc_div_r2'],
        #                t_offset=ntc_cfg['ntc_offset'],
        #                cal_pin=self.adc_cal, cal_duty=2.76/3.3,
        #                samples=15, samples_delay_us=3000)
        oled_cfg = self.config['oled']
        # noinspection PyArgumentList
        spi = machine.SPI(oled_cfg['spi_bus_id'], baudrate=6 * 1024 * 1024, polarity=0, phase=0)
        oled_resume_mode = self.lowpower_phase or machine.reset_cause() not in (0, 1, 6)
        self.oled = SSD1306_SPI_custom(oled_cfg['width'], oled_cfg['height'], spi=spi,
                                       dc=machine.Pin(oled_cfg['dc_pin']),
                                       res=machine.Pin(oled_cfg['res_pin']),
                                       cs=machine.Pin(oled_cfg['cs_pin']),
                                       resume_mode=oled_resume_mode)
        self.oled.contrast(0)
        if not oled_resume_mode:
            self.oled.text('BOOTING', 36, 28)
            self.oled.show()
            self.oled.fill(0)

    def measure(self):
        try:
            self.dht.measure()
            dht_cfg = self.config['dht']
            self.current_temp = ((self.dht.temperature() * dht_cfg['temp_cal_mul']) + dht_cfg['temp_cal_offset'])
            self.current_humidity = ((self.dht.humidity() * dht_cfg['rh_cal_mul']) + dht_cfg['rh_cal_offset'])
        except OSError:
            raise OSError('Couldn\'t read DHT sensor')
        # self.current_temp = self.ntc.read()
        values_averaging = self.config['station'].get('values_average')
        if values_averaging and isinstance(values_averaging, int) and values_averaging > 0:
            self.avg_temp = self.state_recent_avg('recent_temps', values_averaging,
                                                  new_val=self.current_temp)
            self.avg_humidity = self.state_recent_avg('recent_humidities', values_averaging,
                                                      new_val=self.current_humidity)
        else:
            self.avg_temp = self.current_temp
            self.avg_humidity = self.current_humidity

    def display_values(self):
        # noinspection PyUnresolvedReferences
        import writer_minimal as writer

        self.requests = None
        self.dht = None
        self.unload_modules(('requests', 'ntptime', 'flashbdev', 'dht'))

        self.oled.fill(0)

        # noinspection PyUnresolvedReferences
        import futurahvbig
        wri_big = writer.Writer(self.oled, futurahvbig)
        wri_big.set_textpos(0, 0)
        wri_big.printstring('{:.1f}\xB0'.format(self.avg_temp))
        self.oled.scroll(2, -1)
        del wri_big
        del futurahvbig
        self.unload_modules(('futurahvbig',))

        # noinspection PyUnresolvedReferences
        import futurahvsml
        wri_sml = writer.Writer(self.oled, futurahvsml)
        wri_sml.set_textpos(37, 57)
        wri_sml.printstring('{:.0f}%rh'.format(self.avg_humidity))

        self.oled.show()

        del wri_sml
        del futurahvsml
        del writer
        self.unload_modules(('futurahvsml', 'writer'))

    def main(self):
        self.measure()
        self.display_values()

    @staticmethod
    def unload_modules(modules):
        import sys
        import gc
        # noinspection PyUnresolvedReferences
        print('mem_free={} before'.format(gc.mem_free()))
        for mod in modules:
            if mod in sys.modules:
                del sys.modules[mod]
            gc.collect()
            # noinspection PyUnresolvedReferences
            print('Unloaded {}, mem_free={}'.format(mod, gc.mem_free()))

    def before_deepsleep(self):
        if self.lowpower_phase:
            top_text, bottom_text = 'sync', 'in {}m'.format(math.ceil(self.seconds_to_next_sync() / 60))
        else:
            if self.sync_success:
                top_text, bottom_text = 'just', 'syncd'
            else:
                top_text, bottom_text = 'sync', 'fail'
        self.oled.text(top_text, 0, 39)
        self.oled.text(bottom_text, 0, 47)
        self.oled.text('{2:02d}/{1:02d}/{0:04d} {4:02d}:{5:02d}'.format(*self.rtc.datetime()), 0, 57)
        self.draw_battery(self.battery_level(), self.oled, 44, 38, 10, 17)
        self.oled.show()

    def on_low_bat(self):
        print('Low bat: long deepsleep')
        self.oled.fill(0)
        self.oled.text('LOW BAT', 36, 28)
        self.oled.show()
        esp.deepsleep(4_294_967_290)  # almost maxint ~71min

    @staticmethod
    def draw_battery(bat, oled, x0, y0, width, height, cap_pad=None):
        if cap_pad is None:
            cap_pad = height // 5
        inner_height = height - 4
        bar_height = round(bat * inner_height)
        cap_x_start = x0 + cap_pad
        cap_x_end = x0 + width - cap_pad
        cap_y_end = y0 + cap_pad
        body_height = height - cap_pad - 1
        oled.fill_rect(x0 + 2, y0 + 2 + (inner_height - bar_height), width - 4, bar_height, 1)
        oled.fill_rect(x0, y0, cap_pad + 1, cap_pad + 1, 0)
        oled.fill_rect(cap_x_end - 1, y0, cap_pad + 1, cap_pad + 1, 0)
        oled.vline(x0, cap_y_end, body_height, 1)
        oled.hline(x0, y0 + height - 1, width, 1)
        oled.vline(x0 + width - 1, cap_y_end, body_height, 1)
        oled.hline(cap_x_start, y0, width - 2 * cap_pad, 1)
        oled.hline(x0, cap_y_end - 1, cap_pad, 1)
        oled.hline(cap_x_end, cap_y_end - 1, cap_pad, 1)
        oled.vline(cap_x_start - 1, y0, cap_pad - 1, 1)
        oled.vline(cap_x_end, y0, cap_pad - 1, 1)
