"""Run wren on Pier / Harbor benchmarks (e.g. DeepSWE) as an installed agent.

    uv tool install datacurve-pier --with git+https://github.com/oncetrange/wren
    pier run -p deep-swe/tasks --env modal \\
        --agent-import-path wren.integrations.pier_agent:WrenAgent \\
        -m moonshot/kimi-k2.7-code --ae MOONSHOT_API_KEY=$MOONSHOT_API_KEY

Agent kwargs (--ak): version=<git ref of wren to install, default main>,
max_turns=<model calls per task, default 150>.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pier.agents.installed.base import BaseInstalledAgent, CliFlag, with_prompt_template
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist

from wren.integrations import pier_support as support

if TYPE_CHECKING:
    from pier.environments.base import BaseEnvironment
    from pier.models.agent.context import AgentContext


class WrenAgent(BaseInstalledAgent):
    CLI_FLAGS = [CliFlag(kwarg="max_turns", cli="--max-turns", type="int", default=150)]

    @staticmethod
    def name() -> str:
        return "wren"

    def get_version_command(self) -> str | None:
        return f"{support.WREN_BIN} --version"

    def install_spec(self) -> AgentInstallSpec:
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[InstallStep(user=user, run=cmd)
                   for user, cmd in support.install_commands(self._version or "main")],
            verification_command=self.get_version_command(),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        return NetworkAllowlist(domains=support.model_hosts(support.resolve_model(self.model_name)))

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        model = support.resolve_model(self.model_name)
        env = self.build_process_env({model.api_key_env: self._get_env(model.api_key_env)})
        env["WREN_HOME"] = support.WREN_HOME
        await self.exec_as_agent(
            environment,
            command=support.run_command(instruction, model, self.build_cli_flags()),
            env=env,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        run = support.read_run(self.logs_dir)
        if run is None:
            return
        u = run["usage"]
        context.n_input_tokens = u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"]
        context.n_cache_tokens = u["cache_read_tokens"]
        context.n_output_tokens = u["output_tokens"]
        context.cost_usd = run["cost_usd"]
        context.n_agent_steps = run["turns"]
        context.peak_context_tokens = run["peak_context_tokens"] or None
        context.summarization_count = run["compactions"]
        context.metadata = {k: run[k] for k in ("status", "tool_calls", "tool_errors", "duration_s", "session_id")}
