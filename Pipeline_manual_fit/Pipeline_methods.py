#call fit on pipe_best
#returns intermediate X_reduced value
def pipe_fit(pipe, X,y):
	X_scaled = None
	if(pipe_best.steps[0][1]!='passthrough'):
		X_scaled = pipe_best.steps[0][1].fit_transform(X)
	else:
		X_scaled = X
	X_red = None
	if(pipe_best.steps[1][1]!='passthrough'):
		X_red = pipe_best.steps[1][1].fit_transform(X_scaled)
	else:
		X_red = X_scaled
	pipe_best.steps[2][1].fit(X_red, y)
	X_scaled = None
	return X_red

#call predict on pipe_best
#returns intermediate X_reduced value as tuple with predictions
def pipe_predict_proba(pipe, X,y):
	X_scaled = None
	if(pipe_best.steps[0][1]!='passthrough'):
		X_scaled = pipe_best.steps[0][1].fit_transform(X)
	else:
		X_scaled = X
	X_red = None
	if(pipe_best.steps[1][1]!='passthrough'):
		X_red = pipe_best.steps[1][1].fit_transform(X_scaled)
	else:
		X_red = X_scaled
	preds = pipe_best.steps[2][1].predict_proba(X_red, y)
	return X_red, preds 
