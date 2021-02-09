from vaim import *

if __name__ == "__main__":

    # -- set up the gpu env
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # -- instantiate vae object
    vae = VAIM()

    # -- other avilable examples 'sin'
    vae.example = 'x2'
    vae.epochs = 2000
    
    # -- generate the data for toy problems
    if (vae.example == 'sin'):
        x,y = generate_sin_samples(N = 10000, domain = 4)
    elif (vae.example == 'x2'):
        x,y = generate_x2_samples(N = 10000, noise = 0.05, domain = 5)
    else:
        print('wrong model .... ')
        
    # -- split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)   
    history = vae.train(X_train, y_train)
    print('Training is done')
    
    
    # -- load mode with lowest validation error
    vae.model.load_weights(vae.DIR + vae.model_name)
    
    print('Generate and save results ..')
    # -- predict using test samples
    result = vae.predict(vae, X_train, y_test)
    
    
    # -- plot the results
    plot_result(result, X_test, y_test)
    
    # -- get and plot latent z
    Z = vae.get_latent(vae, X_train)
    plot_latent(Z, X_train)
    
    
