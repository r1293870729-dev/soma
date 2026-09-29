# 交付与远端传输状态（2026-09-29快照）

## 已完成

本地全量整理包保留全部6上传原件，解出并组织历史代码、30被动运行/终态checkpoint、12父组合干预与9主动pilot，新增38张逐任务卡、计划、agent提示词、CPU参考和校验/重算/上传脚本。

GitHub仓库原为空的私有仓库。已初始化main；整理内容在独立分支audit/restructure-20260929提交，PR不自动合并。当前远端只应以实际Git树中文件为准，不根据文档引用推断大文件已经存在。

## 尚未完成的远端部分

**不能声称所有附件已经提交GitHub。** 原始PDF/ZIP、历史实验代码工作副本与大型运行资产仍需要从全量本地包批量补传。最大的原ZIP为166770241字节，超过普通Git单文件100MiB限制；本项目配置ZIP/PT与大型干预CSV使用Git LFS。

当前对话GitHub连接支持文本操作，没有本地大文件参数的批量上传入口；本容器直接Git网络连接失败。未用只有指针却没有LFS对象的方式伪装上传，也未把本地ZIP当成GitHub发布资产。

## 本地补传

安装Git/Git LFS并登录GitHub，配置自己的Git提交身份。解压完整本地包而不是轻量文档包。先校验原件与全清单，复核整理分支的完整40位HEAD，确认LFS存储/传输额度。

```sh
python scripts/verify_assets.py --full
python scripts/publish_to_github.py --workdir ../soma_publish --expected-head <实际40位HEAD>
# 上一条仅dry-run；确认后显式执行：
python scripts/publish_to_github.py --workdir ../soma_publish --expected-head <同一HEAD> --execute --push
```

脚本位于全量本地交付包：克隆到全新目录、核对HEAD、秘密扫描、按gitattributes入LFS、提交、上传LFS后非force推送，再验证远端HEAD。失败停止；不改主分支/权限、不删除远端历史。执行回执写reports/local/transfer_receipt.json。

仅有GitHub管理层时，完整历史重算/资产校验会因文件未齐而失败；这是预期保护，不能造空文件使检查变绿。
