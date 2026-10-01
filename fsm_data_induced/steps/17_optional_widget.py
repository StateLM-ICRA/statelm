# @title Interactive tester (the same session remembers clarification)
try:
    import ipywidgets as widgets
    scope=widgets.Dropdown(options=['lab','hospital'],description='Setting:')
    entry=widgets.Text(placeholder='Type an item request',description='Request:',layout=widgets.Layout(width='85%'))
    send=widgets.Button(description='Ask');reset=widgets.Button(description='Reset conversation')
    view=widgets.Output(); live={'session':Session(full_model,'lab')}
    def ask_clicked(_):
        if live['session'].setting!=scope.value: live['session']=Session(full_model,scope.value)
        with view:
            reply=live['session'].ask(entry.value)
            print('You:',entry.value); print('Robot:',reply['text'])
            print('Action:',reply['action'],'| Drawer:',reply.get('drawer'))
        entry.value=''
    def reset_clicked(_):
        live['session']=Session(full_model,scope.value);view.clear_output()
    send.on_click(ask_clicked);reset.on_click(reset_clicked)
    display(widgets.VBox([scope,entry,widgets.HBox([send,reset]),view]))
except ImportError:
    print('Widgets unavailable. Use: session = Session(full_model, "lab"); session.ask("Where is tape?")')