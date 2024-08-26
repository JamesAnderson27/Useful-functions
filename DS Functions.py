## Preprocessing

def detect_na(df):
    # Input
        # df
    # Output
        # A visualization of the count of NaN or Null values per column
    out=[]
    for col in df.columns:
        count = sum(df[col].isnull())
        out.append(count)
        
    for i, count in enumerate(out):
        print('NaN Values ----  '+str(df.columns[i])+' \n',count)


## Central Limit Theorum Function


def sampler(data, num_trials):
 # Input
 	# data — a distribution
 	# num_trials — n means collected
 # Output
 	# out — a distribution of sample means

    out=[]
    for i in range(num_trials):
        sample = np.random.choice(data,10) # controls the spread of the distribution
        out.append(np.mean(sample))
    
    return out


## Bootstrapping / Permutation Functions


def bootstrapReplicate(data,func):
# Inputs
	# data - distribution
	# func - a measure of centrality functions (mean,med,mode)
# Outputs 
	# A single bootstrapped datapoint 

    bs_sample = np.random.choice(data,len(data))
    return func(bs_sample)
    


def bootstrapSample(data,func,size):
# Inputs
	# size — the length of the bootstrap sample
	# data - distribution
	# func - a measure of centrality functions (mean,med,mode)
# Outputs
	# A single bootstrap sample of n=size

    bs_replicate = np.empty(size)
    
    for i in range(size):
        bs_replicate[i]=bootstrapReplicate(data,func)
        
    return bs_replicate



def draw_bs_pairs_linreg(x, y, size=1):
	# Inputs
		# x — a distribution
		# y — a distribution
		# size — the chosen number of bootstrapped slope & intercepts
	# outputs
		# bs_slope_reps — a distributions (n=size) of bootstrapped slopes
		# bs_intercept_reps — a distribution (n=size) of bootstrapped intc.

    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps


## BROKEN
def perm_diff(sample_1, sample_2, num_iter):
    # Inputs	
    	# sample_1 — a distribution
    	# sample_2 — a distribution
    	# num_iter — how many permuted sample differences to output
    # Outputs
    	# perm_replicates — distribution of diffs between 2 samples


    # initialize the list for the test statistic replicate
    perm_replicates = []
    # iterate for the specified number of iterations
    for i in range(num_iter):
        
        # concatenate the two samples into one
        samples_app = sample_1 + sample_2
        
        # permute the entire appended set (making this complete combined resampling WITHOUT REPLACEMENT)
        samples_perm = np.random.permutation(samples_app)
        
        # create the hypothesized samples by:
        # pretending that the first len(sample_1) elements is the first sample
        sample_1_hyp = samples_perm[:(len(sample_1))]
        sample_2_hyp = samples_perm[(len(sample_1)):]

        #  and the rest is the second sample
        # compute the test statistic replicate and append it to the list of permutation replicates
        diff = np.mean(sample_1_hyp) - np.mean(sample_2_hyp)
        perm_replicates.append(diff)
        
    return perm_replicates # if disrtibutions are sig dif (should be 0)


## BROKEN
def permDiff(sample_1,sample_2,n_iter):
    
    diff_ = []
    for i in range(n_iter):
        
        combined = sample_1+sample_2 # combine samples
        
        permuted = np.random.permutation(combined) # take permutation
        
        p_sample_1 = permuted[0:len(sample_1)]# find perm_sample means
        p_sample_2 = permuted[len(sample_1):]
        
        diff = np.mean(p_sample_1)-np.mean(p_sample_2)
        
        diff_.append(diff) # add difference to output
        
    return(diff_)


## Visualization Functions
def donut_graph(font="Times New Roman",
				names, # list of names per category
				size, # list of sizes per category 
				title=None # title of the graph

	)
	plt.rcParams["font.family"] = font

	f,ax=plt.subplots(figsize=(8,7))  

	# create data
	names = ['Yes (56%)', 'No (33%)', 'Somewhat (11%)']
	size = [56,36,11]
	 
	# Create a circle at the center of the plot
	my_circle = plt.Circle( (0,0), 0.7, color='white')

	# Give color names
	ax.pie(size, labels=names,
	        wedgeprops = { 'linewidth' : 5, 'edgecolor' : 'white' },
	       textprops={'fontsize': 15},
	        colors=['steelblue','lightblue','dodgerblue'])
	p = plt.gcf()
	p.gca().add_artist(my_circle)

	ax.set_title(title,
	            fontsize=19)


def ecdf(array):
    # input — (np array)
    # output — a sorted array as the x, and a scaled array as the y
    
    x=np.sort(array)
    y=np.arange(1,len(x)+1) / len(x)
    
    return x,y


## Probability Functions 

def coinFlip():
    # Out (Bool): True for heads, False for tails
    flip=np.random.random()
    if flip>=.5:
        return True
    if flip<.5:
        return False



def flipTrials(num_trials,num_success):
    # in (int): num of trials
    # out (float): the probability of getting num_success heads per num_success flips 
        # given (num_trials) number of trials
    
    all_heads=0
    
    for i in range(num_trials):
        # 1 single trial
        trial=[coinFlip() for i in range(num_success)]
        if np.sum(trial)==num_success:
            all_heads+=1
    return all_heads/num_trials



def flipTrials(num_trials):
    # in (int): num of trials
    # out (float): the probability of getting num_success heads per num_success flips 
        # given (num_trials) number of trials
    all_heads=0
    for i in range(num_trials):
        trial = np.random.random(size=num_success)<.5
        n_head = np.sum(trial)
        if n_head==num_success:
            all_heads+=1
    return all_heads/num_trials

