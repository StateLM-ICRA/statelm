# @title Examples and manual clarification
demo_rows=[]
for question in ['Where is SD card?', 'Where is the sd crad?', 'Where is the red tape?', 'I need a pen.', 'Tell me a joke.']:
    reply=full_model.request(question,'lab')
    demo_rows.append({'question':question,'action':reply['action'],'answer':reply['text'],'drawer':reply.get('drawer')})
display(pd.DataFrame(demo_rows))
demo_session=Session(full_model,'lab')
for question in ['Where is tape?', 'Blue.']:
    print('You:',question); print('Robot:',demo_session.ask(question)['text'])