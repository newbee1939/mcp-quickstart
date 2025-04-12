import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

# .envファイルから環境変数（ANTHROPIC_API_KEY）を読み込む
load_dotenv()

class MCPClient:
    """
    Model Context Protocol (MCP)クライアントクラス

    MCP対応のサーバーと通信し、ユーザーのクエリを処理するためのクライアント。
    Claude APIを使用してLLMとの対話を管理し、サーバー上のツールを呼び出す機能を提供します。
    """
    def __init__(self):
        # セッションとクライアントオブジェクトの初期化
        self.session: Optional[ClientSession] = None  # MCPサーバーとのセッション
        self.exit_stack = AsyncExitStack()  # 非同期リソース管理用のコンテキストマネージャ
        self.anthropic = Anthropic()  # Anthropic APIクライアント

    async def connect_to_server(self, server_script_path: str):
        """
        MCPサーバーに接続する

        指定されたスクリプトパスに基づいてサーバーを起動し、接続を確立します。
        Python(.py)またはNode.js(.js)のサーバースクリプトに対応しています。

        Args:
            server_script_path: サーバースクリプトのパス（.pyまたは.js）
        """
        # スクリプトの種類を確認（PythonかNode.jsか）
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        # スクリプト種類に応じたコマンドを選択
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None  # 環境変数の追加設定なし
        )

        # 標準入出力ベースのトランスポート接続を確立
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        # クライアントセッションを初期化
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # サーバーセッションの初期化
        await self.session.initialize()

        # 利用可能なツールのリストを取得して表示
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """
        Claude APIとMCPサーバーのツールを使用してクエリを処理する

        ユーザークエリをClaudeに送信し、Claudeが必要に応じてツールを呼び出し、
        結果を処理して最終的な応答を生成します。

        Args:
            query: ユーザーから受け取ったクエリ文字列

        Returns:
            最終的な応答テキスト
        """
        # 会話履歴を初期化（ユーザークエリを最初のメッセージとして設定）
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # サーバーから利用可能なツールのリストを取得
        response = await self.session.list_tools()
        # Claudeに提供するツール情報を形式化
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # 初回のClaude API呼び出し
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",  # 使用するモデルバージョン
            max_tokens=1000,  # 生成する最大トークン数
            messages=messages,  # 会話履歴
            tools=available_tools  # 利用可能なツール情報
        )

        # レスポンスの処理とツール呼び出しの処理
        final_text = []  # 最終的な応答テキストを格納するリスト

        # コンテンツの種類に応じた処理
        for content in response.content:
            if content.type == 'text':
                # テキスト応答の場合はそのまま追加
                final_text.append(content.text)
            elif content.type == 'tool_use':
                # ツール呼び出しの場合
                tool_name = content.name  # 呼び出すツール名
                tool_args = content.input  # ツールに渡す引数

                # サーバー上でツールを実行
                result = await self.session.call_tool(tool_name, tool_args)
                # ツール呼び出しの情報をログとして追加
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # ツール結果を含めた会話の継続
                if hasattr(content, 'text') and content.text:
                    messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                # ツールの実行結果をユーザーメッセージとして追加
                messages.append({
                    "role": "user",
                    "content": result.content
                })

                # ツール実行結果を踏まえて再度Claude APIを呼び出し
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                )

                # 新しい応答テキストを追加
                final_text.append(response.content[0].text)

        # 収集したすべてのテキストを連結して返す
        return "\n".join(final_text)

    async def chat_loop(self):
        """
        対話型チャットループを実行

        ユーザー入力を受け取り、process_queryメソッドで処理し、
        結果を表示するループ処理を行います。'quit'と入力されるまで継続します。
        """
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                # ユーザー入力を取得
                query = input("\nQuery: ").strip()

                # 終了条件の確認
                if query.lower() == 'quit':
                    break

                # クエリを処理して応答を取得
                response = await self.process_query(query)
                # 応答を表示
                print("\n" + response)

            except Exception as e:
                # エラー処理
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """
        リソースのクリーンアップ

        AsyncExitStackを使用して確保したすべての非同期リソースを適切に解放します。
        """
        await self.exit_stack.aclose()

async def main():
    """
    メインエントリーポイント

    コマンドライン引数からサーバースクリプトのパスを取得し、
    MCPクライアントを初期化して実行します。
    """
    # コマンドライン引数のチェック
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    # クライアントの初期化と実行
    client = MCPClient()
    try:
        # サーバーに接続
        await client.connect_to_server(sys.argv[1])
        # チャットループを開始
        await client.chat_loop()
    finally:
        # 終了時に必ずリソースをクリーンアップ
        await client.cleanup()

if __name__ == "__main__":
    import sys
    # 非同期メイン関数の実行
    asyncio.run(main())
