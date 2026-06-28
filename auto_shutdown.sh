#!/bin/bash
while pgrep -f "train_test_cb1_grid.py" > /dev/null; do
    sleep 60
done
echo "$(date): 训练结束，即将关机" >> /home/ubuntu/project_ciah/grid_full.log
sudo shutdown -h +1
