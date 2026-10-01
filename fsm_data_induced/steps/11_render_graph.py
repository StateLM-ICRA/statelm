# @title Display and export the induced graph
graph = save_graph(full_model, RESULTS / 'learned_fsm.dot')
print(full_model.graph_stats)
if shutil.which('dot'):
    svg = graph.pipe(format='svg')
    (RESULTS / 'learned_fsm.svg').write_bytes(svg)
    display(SVG(svg))
elif hashlib.sha256((RESULTS / 'learned_fsm.dot').read_bytes()).hexdigest() == (DATA / 'verified_graph.sha256').read_text():
    # Exact same DOT: use its included verified rendering on hosts without Graphviz.
    svg = (DATA / 'verified_graph.svg').read_bytes()
    (RESULTS / 'learned_fsm.svg').write_bytes(svg)
    display(SVG(svg))
else:
    print('DOT graph exported. Install Graphviz to render this changed graph.')
    display(pd.DataFrame(full_model.artifact()['states']))
print('Double circles accept a request. After acceptance, the learned table supplies the drawer and the learned template supplies wording.')