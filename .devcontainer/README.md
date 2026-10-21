# 开发容器

本项目包含一个 [开发容器](https://containers.dev/)，可让你将容器用作功能完整的开发环境。

你可以使用本文件夹中的开发容器配置来构建和运行应用程序，而无需在本地安装任何工具！你可以在 [GitHub Codespaces](https://github.com/features/codespaces) 或 [VS Code 开发容器扩展](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 中使用它。

## GitHub Codespaces

[![在 GitHub Codespaces 中打开](https://github.com/codespaces/badge.svg)](https://codespaces.new/langchain-ai/langchain)

你可以使用上方的按钮，或按照以下步骤在 Codespace 中打开此仓库：

1. 在 <https://github.com/langchain-ai/langchain> 顶部点击 **Code** 下拉菜单。
1. 点击 **Codespaces** 标签页。
1. 点击 **在 master 上创建 Codespace**。

有关更多信息，请参阅 [GitHub 文档](https://docs.github.com/en/free-pro-team@latest/github/developing-online-with-codespaces/creating-a-codespace#creating-a-codespace)。

## VS Code 开发容器

[![在开发容器中打开](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/langchain-ai/langchain)

> [!NOTE]
> 如果你点击上方链接，将会打开主仓库（`langchain-ai/langchain`），而不是你的本地克隆仓库。如果你仅想运行和测试库，这没有问题；但如果你想贡献代码，可以使用下方链接，并替换为你的用户名和克隆的仓库名称：

```txt
https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/&lt;YOUR_USERNAME&gt;/&lt;YOUR_CLONED_REPO_NAME&gt;
```

这样你就能获得一个本地克隆的仓库，用于贡献并创建拉取请求。

如果你已安装 VS Code 和 Docker，可以使用上方按钮开始使用。该操作将通过 VS Code 自动安装所需的开发容器扩展，将源代码克隆到容器卷中，并启动一个开发容器供你使用。

或者，你也可以按照以下步骤，使用 VS Code 开发容器扩展在此仓库中打开容器：

1. 如果这是你首次使用开发容器，请确保你的系统满足先决条件（例如已安装 Docker），详见 [入门步骤](https://aka.ms/vscode-remote/containers/getting-started)。

2. 打开本地克隆的代码：

   - 将此仓库 fork 并克隆到你的本地文件系统。
   - 按下 <kbd>F1</kbd>，然后选择 **Dev Containers: 在容器中打开文件夹...** 命令。
   - 选择此文件夹的克隆副本，等待容器启动后即可尝试使用！

你可以在 [开发容器文档](https://code.visualstudio.com/docs/devcontainers/containers) 中了解更多信息。

## 技巧与提示

- 如果你在容器和 Windows 系统中同时使用同一个仓库文件夹，建议保持一致的换行符格式（否则 SCM 视图中可能会看到数百个更改）。本仓库根目录下的 `.gitattributes` 文件会禁用换行符转换，从而防止此类问题。更多详情请参见 [技巧与提示](https://code.visualstudio.com/docs/devcontainers/tips-and-tricks#_resolving-git-line-ending-issues-in-containers-resulting-in-many-modified-files)。
- 如果你想查看本开发容器所使用的镜像内容，可以前往 [devcontainers/images](https://github.com/devcontainers/images/tree/main/src/python) 仓库进行查阅。
