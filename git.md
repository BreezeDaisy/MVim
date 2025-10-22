# 修改远程仓库地址为 HTTPS
git remote set-url origin https://github.com/BreezeDaisy/MVim.git

# 推送（若需要代理，直接用 HTTPS 代理参数）
https_proxy=http://172.28.238.143:6500 git push -u origin main

# SSH 协议（需配置 SSH 密钥）
git remote set-url origin git@github.com:git@github.com:BreezeDaisy/MVim.git

# git 流程

## 开始工作前
### 拉取远程 main 分支的最新更新（当前分支需为 main）
git pull origin main

#### 若需要代理
https_proxy=http://172.28.238.143:6500 git pull origin main

## 开发过程中
git status  # 查看工作区变动
git diff    # 查看具体修改内容（可选）

### 暂存修改
git add 文件名1 文件名2  # 提交指定文件
或提交所有修改（除 .gitignore 排除的内容）
git add .

### 提交修改
git commit -m "功能/修复：具体说明，例如“新增用户登录模块”或“修复首页加载bug”"

## 开发完成
### 推送本地当前分支到远程 main 分支
git push origin main

### 若需要代理
https_proxy=http://172.28.238.143:6500 git push origin main

## 撤销未提交的修改
git checkout -- 文件名  # 撤销单个文件的修改（未 add 状态）
git reset HEAD 文件名   # 撤销已 add 到暂存区的文件