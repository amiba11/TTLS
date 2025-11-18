# TTLS

代码基于recbole实现，需先下载：https://github.com/RUCAIBox/RecBole

1.backbone 新增对模型原损失的修改
2.recbole_data  新增对rebole数据划分部分的修改。
3.recbole_train  新增evaluate_ttt( TTL 的 last item 熵减优化 )
4.merge_model 新增模型合并，由quick_run启动
5.run_fine_tuning 二次微调，由quick_run启动

上述可以直接替换recbole原始文件。
