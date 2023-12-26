        # Iterate over the features
        for feature in range(complete_points.shape[1] - 1):

            # Iterate over the incomplete data points
            for i in range(incomplete_points.shape[0]):

                # If the feature value is NaN, find the K-nearest neighbors and fill in the missing value with the average value of the corresponding feature in the K-nearest neighbors
                if np.isnan(clean_points[i, feature]):

                    # Get the complete data points with the same class label as the incomplete data point
                    complete_points_with_same_label = complete_points[complete_points[:, -1] == clean_points[i, -1], :]

                    # Calculate the pairwise distances between the incomplete data point and the complete data points
                    distances = self.pairwise_dist(clean_points[i, :-1], complete_points_with_same_label[:, :-1])

                    # Find the K-nearest neighbors
                    nearest_neighbors = np.argsort(distances, axis=0)[:K]

                    # Check if the `complete_points_with_same_label` array has a third dimension
                    if complete_points_with_same_label.ndim == 3:
                        # Sum the elements of the array along axis 2
                        clean_points[i, feature] = np.sum(complete_points_with_same_label[nearest_neighbors, feature, :], axis=2).mean()
                    else:
                        # Sum the elements of the array without specifying an axis
                        clean_points[i, feature] = np.sum(complete_points_with_same_label[nearest_neighbors, feature]).mean()

        return clean_points
    
    
    
            '''
        1) Iterate feature 
        2) Iter incomplete 
        3) NaN ==> knn to fill it 
        4) get incom dp 
        5) distance b/w comp and incomp 
        6) knn 
        7) check dim
        '''
