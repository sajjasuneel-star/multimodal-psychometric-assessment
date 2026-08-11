TRAITS=['engagement','collaboration_quality','socio_emotional_presence','cognitive_participation','participation_balance']

def generate_intervention(traits,uncertainty,threshold=.18):
    d=dict(zip(TRAITS,[float(x) for x in traits])); u=float(max(uncertainty))
    messages=[]
    if u>threshold:
        messages.append('Instructor review recommended because predictive uncertainty is high.')
    if d['participation_balance']<.33:
        messages.append('Encourage contributions from less-active group members.')
    if d['engagement']<.33:
        messages.append('Introduce a short engagement-oriented collaborative prompt.')
    if d['collaboration_quality']<.33:
        messages.append('Prompt clarification, repair, and consensus-building dialogue.')
    if d['cognitive_participation']<.33:
        messages.append('Ask learners to justify reasoning and explain intermediate steps.')
    if d['socio_emotional_presence']<.33:
        messages.append('Encourage supportive responses and constructive disagreement management.')
    if not messages:
        messages.append('No immediate intervention; continue monitoring the collaborative trajectory.')
    return messages
