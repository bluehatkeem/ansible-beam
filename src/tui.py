from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Button, Input, Static
from textual.reactive import reactive
from textual.message import Message
#from ansible_resource_generator import AnsibleResourceGenerator
from beam import AnsibleResourceGenerator

class AnsibleApp(App):
    CSS_PATH = "styles.css"

    class GenerateResource(Message):
        def __init__(self, resource_type: str, description: str) -> None:
            self.resource_type = resource_type
            self.description = description
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Ansible Resource Generator", id="title"),
            Input(placeholder="Enter resource type (playbook/role)", id="resource_type"),
            Input(placeholder="Enter description", id="description"),
            Button("Generate", id="generate_button"),
            id="main_container"
        )
        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate_button":
            resource_type = self.query_one("#resource_type", Input).value
            description = self.query_one("#description", Input).value
            await self.post_message(self.GenerateResource(resource_type, description))

    async def on_generate_resource(self, message: GenerateResource) -> None:
        generator = AnsibleResourceGenerator()
        new_resource = generator.generate_resource(message.resource_type, message.description)
        self.query_one("#main_container", Container).mount(Static(f"Generated Resource:\n{new_resource}"))

if __name__ == "__main__":
    app = AnsibleApp()
    app.run()
