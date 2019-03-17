# -*- coding: utf-8 -*-


def sec_to_time(sec):
    return int(sec / 3600), int((sec % 3600) / 60), int(sec % 60)
