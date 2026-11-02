from pathlib import Path


class TextLogger:
    """简单文本日志工具。

    用于把训练、验证或测试过程中的文字信息同时输出到控制台和日志文件。
    """

    def __init__(self, path: Path):
        """初始化日志文件路径，并自动创建父目录。"""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, text: str) -> None:
        """写入一行日志：先打印到终端，再追加保存到文本文件。"""
        print(text)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(text + "\n")
