"""DAQmx scaling functions"""
import math


def rtdscale(voltage, excitation, r0, a, b, c, r_lead, configuration, method):
    """Convert measured voltage into temperature

    Args:
        voltage (float): measured voltage across RTD resistance
        excitation (float): excitation current through RTD
        r0 (float): RTD nominal resistance at 0 °C
        a, b, and c (float): Callendar-Van Dusen coefficients
        r_lead (float): lead resistance
        configuration (int): RTD 2/3/4-wire configuration
        method (str): DAQmx_rtdScale or Newton-Raphson

    Returns:
        (tuple): calculated temperature in °C, # iterations
    """
    r = voltage / excitation  # RTDResistance
    if configuration == 3:
        r = r - r_lead
    elif configuration == 2:
        r = r - 2 * r_lead
    loop = 0
    if method == "DAQmx_rtdScale":
        coefficients = [(r / r0 - 1) / a, 0, -b / a, 100 * c / a, -c / a]
        if r0 - r > 0:
            t0 = 0.0
            t1 = -100.0
            tolerance = 0.1
            loop_limit = 500
            while abs(t1 - t0) > tolerance and loop < loop_limit:
                t0 = t1
                t1 = coefficients[0] + t0 * (
                    t0
                    * (coefficients[2] + t0 * (coefficients[3] + t0 * coefficients[4]))
                )
                loop += 1
            t = t1
        else:
            num = math.sqrt(r0 * (r0 * a ** 2 - 4 * r0 * b + 4 * b * r))
            den = 2 * r0 * b
            t = num / den - a / (2 * b)
    # elif method == "Newton-Raphson":
    #     num = math.sqrt(r0 * (a ** 2 * r0 - 4 * b * r0 + 4 * b * r))
    #     den = 2 * b * r0
    #     t = num / den - a / (2 * b)
    #     if t < 0:
    #         tolerance = 1e-6
    #         t0 = t
    #         num = r0 * (1 + a * t + b * t ** 2 + (t - 100) * c * t ** 3) - r
    #         den = r0 * (a + 2 * b * t + 4 * c * t ** 3 - 300 * c * t ** 2)
    #         t1 = t0 - num / den
    #         loop_limit = 500
    #         while abs(t1 - t0) > tolerance and loop < loop_limit:
    #             t0 = t1
    #             num = r0 * (1 + a * t0 + b * t0 ** 2 + (t0 - 100) * c * t0 ** 3) - r
    #             den = r0 * (a + 2 * b * t0 + 4 * c * t0 ** 3 - 300 * c * t0 ** 2)
    #             t1 = t0 - num / den
    #             loop += 1
    #         t = t1
    else:
        raise ValueError(f"{method} not valid")
    return (t, loop)
