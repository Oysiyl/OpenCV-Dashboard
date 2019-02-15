#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 18:22:53 2019.

@author: dmitriy
"""
from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()


@sched.scheduled_job('interval', minutes=3)
def timed_job():
    """Select time interval."""
    print('This job is run every three minutes.')


@sched.scheduled_job('cron', day_of_week='mon-fri', hour=17)
def scheduled_job():
    """Schedule a job."""
    print('This job is run every weekday at 5pm.')


sched.start()
