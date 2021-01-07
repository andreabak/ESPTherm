import machine
import math

import utime


class NTC:

    ABS0_C = -273.15
    ADC_MAX = 1023

    def __init__(self, pin, ntc_nom_r, ntc_nom_t, ntc_bcoeff, vdiv_r2,
                 cal_pin=None, cal_duty=None, t_offset=0.0, samples=5, samples_delay_us=500):
        self.adc = machine.ADC(pin)
        self.ntc_nom_r = ntc_nom_r
        self.ntc_nom_t = ntc_nom_t
        self.ntc_bcoeff = ntc_bcoeff
        self.vdiv_r2 = vdiv_r2
        if cal_pin is not None and cal_duty is None:
            raise AttributeError('If specifying cal_pin, a cal_duty value (0.0-1.0) must be specified too')
        self.cal_pin = cal_pin
        self.cal_duty = cal_duty
        self.temp_offset = t_offset
        self.samples = samples
        self.samples_delay_us = samples_delay_us

    def read(self, samples=None, samples_delay_us=None):
        def adc_read_sleep():
            nonlocal self, samples_delay_us
            v = self.adc.read()
            utime.sleep_us(samples_delay_us)
            return v

        if samples is None:
            samples = self.samples
        if samples_delay_us is None:
            samples_delay_us = self.samples_delay_us

        cal_adjust = 1.0
        if self.cal_pin is not None:
            self.cal_pin.on()
            utime.sleep_us(samples_delay_us)
            cal_value = adc_read_sleep()
            self.cal_pin.off()
            utime.sleep_us(samples_delay_us)
            cal_adjust = self.cal_duty / (cal_value / self.ADC_MAX)

        raw = float(sum(adc_read_sleep() for _ in range(samples))) / samples
        raw_duty = (raw / self.ADC_MAX) * cal_adjust
        ntc_r = self.vdiv_r2 * ((1.0 / raw_duty) - 1.0)
        temp_k = 1.0 / ((math.log(ntc_r / self.ntc_nom_r) / self.ntc_bcoeff) + (1.0 / (self.ntc_nom_t - self.ABS0_C)))
        temp_k += self.temp_offset
        temp_c = temp_k + self.ABS0_C
        return temp_c
