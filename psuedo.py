



Need a script to generate around 100 ish coordinates in an excel file?
Need a way to generate the distance matrix {

    Maybe use this code like how they do it here

            N = [i for i in range(1, n+1)] # total customers
            V = [0] + N # add depot

            A = [(i, j) for i in V for j in V if i != j] # matrix distance

            # Using direction API to do distance calculation
            c = {(i, j): np.hypot(xc[i]-xc[j], yc[i]-yc[j]) for i, j in A} #distance calculation

            # Save distance matrix 
            with open(r'distance_matrix_cvrp.txt','w+') as f:
                f.write(str(c))
}

Then generate an optimization technique(s) and evaluate their performance
Visualize it on folium, this time base it in a suburb
Use Open Route Service for routes on roads


Psuedo for MRA script{
    setup the depot coordinate. either by reading in a json or whichever
    setup the amount of nodes/customer
    combine the depot and node coordinate into a seperate longitude and lattitude list
    create index for each node/customer
    create a depot list
    create a distance matrix 
    calculate the distance using euclidean distance
    save the distance matrix 
}


top left :
Latitude: -37.7927
Longitude: 145.0225

bottom right :
Latitude: -37.8501
Longitude: 145.1119