from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from tasks.get_notice import crawl_notice

scheduler = BackgroundScheduler(timezone="Asia/Shanghai")


def start_scheduler():
    scheduler.add_job(
        crawl_notice,
        trigger=CronTrigger(hour=10, minute=6),  # 每日 09:45 执行
        id="crawl_notice",
        replace_existing=True,
        max_instances=1,      # 最多一个实例
        coalesce=True,        # 错过多次只执行一次
        misfire_grace_time=3600,  # 最多补执行1小时内错过的任务
    )

    scheduler.start()