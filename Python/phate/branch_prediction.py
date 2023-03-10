import numpy as np
import graphtools
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin
import scprep, scipy as sp, phate

class BranchPointPredictor(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        phate_op: phate.PHATE, # a trained PHATE operator
        extrema_percentile:float = 50, # percentile to mask when calculating extrema
        diffusion_iterations:int=20 # number of iterations to diffuse
    ):
        
        self.phate_op = phate_op
        self.extrema_percentile = extrema_percentile
        self.diffusion_iterations = diffusion_iterations
                      
    def fit(self, X, y=None):
        self.diffuse_dirac_for_end_points()
        self.assign_branches(X)
        self.plot_branchs(X)
        self.plot_branch_classes(X)
        return self
    
    def transform(self, X):
        return self.classes
    

    # NOTE: these two properties are for convenience of the developer
    # and they just expose the underlying PHATE operator values
    @property
    def diff_op(self):
        try:
            return self.phate_op.diff_op
        except AttributeError:
            return None

    @property
    def optimal_t(self):
        try:
            return self.phate_op.optimal_t
        except AttributeError:
            return None
    

    # NOTE: listing all properties up top for readability
    @property
    def dmap(self):
        '''
        Returns the diffusion map calculated from the diffusion operator
        '''
        try:
            return self._dmap
        except AttributeError:        
            self._calc_dmap()            
            return self._dmap 

    @property
    def n_use(self):
        '''
        The number of eigenvectors in the diffusion map to use
        for downstream analyses
        '''
        try:
            return self._n_use
        except AttributeError:        
            self._calc_num_to_consider()            
            return self._n_use 

    @property
    def most_distinct_points(self):
        '''
        The most distinct points **prior** to downstream analysis.
        These are the extrema.
        '''
        try:
            return self._most_distinct_points
        except AttributeError:        
            self._calc_extrema()          
            return self._most_distinct_points
       
    @property
    def is_landmarked(self):
        '''
        Whether or not the graph in the PHATE operator is a Landmark Graph
        which matters when reconstructing class labels
        '''
        return isinstance(self.phate_op.graph, graphtools.graphs.kNNLandmarkGraph)
    
    # NOTE: these two properties are for handling reconstruction from the landmark operator
    # back to the original data space.
    @property
    def pmn(self):
        try:
            return self.phate_op.graph.transitions
        except Exception:        
            return None
        
    @property
    def pnm(self):
        try:
            return self.phate_op.graph._data_transitions()
        except Exception:        
            return None
        
    @property
    def n_rows(self):
        return self.pmn.shape[0] if self.pmn is not None else self.diff_op.shape[0]
    
    @property
    def nn_dist(self):
        '''
        Nearest Neighbor distance matrix calculated on diffusion operator
        '''
        try:
            return self._nn_dist
        except AttributeError:
            self._knn_on_diff_op()
            return self._nn_dist
        
    @property
    def nn_idxs(self):
        '''
        Nearest Neighbor indicies calculated on diffusion operator
        '''
        try:
            return self._nn_idxs
        except AttributeError:
            self._knn_on_diff_op()
            return self._nn_idxs
        
    @property
    def n_nbrs(self):
        try:
            return self._n_nbrs
        except AttributeError:
            self.max_likelihood_pointwise_dimensionality_est()
            return self._n_nbrs
    
    @property
    def nbrs_dim_est(self):
        try:
            return self._nbrs_dim_est
        except AttributeError:
            self.max_likelihood_pointwise_dimensionality_est()
            return self._nbrs_dim_est
    
    @property
    def most_distinct_points_adjusted(self):
        try:
            return self._most_distinct_points_adjusted
        except AttributeError:
            self.max_likelihood_pointwise_dimensionality_est()
            return self._most_distinct_points_adjusted

    @property
    def classes(self):
        '''
        Branch class labels
        '''
        try:
            return self._classes
        except AttributeError:
            self.diffuse_dirac_for_end_points()
            return self._classes
        
    @property
    def branch_classes(self):
        try:
            return self._branch_classes
        except AttributeError:
            self.assign_branches(self.phate_op.X)
            return self._branch_classes

    @property
    def branch_points(self):
        try:
            return self._branch_points
        except AttributeError:
            self.diffuse_dirac_for_end_points()
            return self._branch_points
    
    # NOTE: sets property dmap
    def _calc_dmap(self, t=None):
        if t is None:
            t = self.optimal_t

        evals, evecs = np.linalg.eig(self.diff_op)
                
        # sort eigenvectors in descending order
        idx = np.abs(evals).argsort()[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        # do diffusion        
        evals = np.power(evals, self.optimal_t)
        evecs = evecs.dot(np.diag(evals))

        self.evals = evals
        self.evecs = evecs
        self._dmap = evecs   
        return evecs

    # NOTE: sets property n_use      
    def _calc_num_to_consider(self):
        dmap = self.dmap
        evals = self.evals
        
        # Number of eigenvectors (~ dimensions) to consider.
        dmap_diff = evals - np.roll(evals, 1)
        
        n_evecs = 1
        # Increase the number of eigenvectors until 
        while (dmap_diff[n_evecs + 1] > 2 * dmap_diff[n_evecs]):
            n_evecs += 1
        
        self._n_use = n_evecs
        return n_evecs
    
    # NOTE: sets property most_distinct_points
    def _calc_extrema(self):
        # NOTE: these functions are equivalent, but
        # v2 is used in latest version on GitHub and
        # although v1 looks cleaner
        return self.__calc_extrema_v2()
        return self.__calc_extrema_v1()
    
    def __calc_extrema_v1(self):
        dmap = self.dmap

        # Ignore first (trivial) eigenvector
        dmap = dmap[:, 1:].copy()

        # Mask lower 50% abs val
        lower_half_abs = np.percentile(np.abs(dmap), self.extrema_percentile)
        dmap[np.abs(dmap) < lower_half_abs] = 0

        max_idxs = dmap.argmax(axis=0)
        min_idxs = dmap.argmin(axis=0)
        extrema_idxs = np.unique(np.hstack((max_idxs, min_idxs)))
        self._most_distinct_points = extrema_idxs
        return extrema_idxs

    def __calc_extrema_v2(self):
        # Find the extremas (min and max) of the considered eigenvectors.
        # Keep them in the order of the eigenvalues by weaving min and max values.
        # min_eigs = phate_op_eigvecs[:,1:n_eigvecs+1].argmin(0)
        # max_eigs = phate_op_eigvecs[:,1:n_eigvecs+1].argmax(0)
        # combined_eigs = np.empty((min_eigs.size + max_eigs.size,), dtype=min_eigs.dtype)
        # combined_eigs[0::2] = min_eigs
        # combined_eigs[1::2] = max_eigs

        # Remove duplicates.

        # for e in combined_eigs:
        #     if e not in most_distinct_points:
        #         most_distinct_points.append(e)

        n_consider = self.n_use
        dmap = self.dmap

        most_distinct_points = []

        # Always skip the first trivial eigenvector
        for i in np.arange(n_consider):
            cur_eigvec = np.copy(dmap[:,i+1])
            # Sometimes the eigvectors are skewed towards one side (much more possitive values than negative 
            # values and vice versa). This part ensures only the extrema on the more significant side is taken.            
            lower_half_abs = np.percentile(np.abs(cur_eigvec), self.extrema_percentile)
            cur_eigvec[np.abs(cur_eigvec) < lower_half_abs] = 0

            max_eig = np.argmax(cur_eigvec)
            min_eig = np.argmin(cur_eigvec)

            if cur_eigvec[max_eig] > 0 and max_eig not in most_distinct_points:
                most_distinct_points.append(max_eig)
            
            if cur_eigvec[min_eig] < 0 and min_eig not in most_distinct_points:
                most_distinct_points.append(min_eig)

        most_distinct_points = np.array(most_distinct_points)
        self._most_distinct_points = most_distinct_points
        return most_distinct_points        

        
    def _knn_on_diff_op(self):
        # NOTE: using KNN on diff_map is not invertable
        # i.e. need to revert landmark graph here!
        dmap = self.dmap

        #######################
        # INTRINSIC DIMENSION #
        #######################

        # Based on maxLikPointwiseDimEst() of this R package.
        # https://cran.r-project.org/web/packages/intrinsicDimension/README.html

        # Up to 100 dimensions of diffusion maps, 
        # raised to the same power as tdetermined by PHATE.
        dm_dims = min(self.diff_op.shape[1], 100) # NOTE: oroginaly was data.shape[1]
        diff_map = dmap[:,:dm_dims]
        # diff_map = diff_map.dot(np.diag(np.power(phate_op_eigvals[:dm_dims], 11)))
        if self.is_landmarked:
            diff_map = self.phate_op.graph.interpolate(diff_map)
    
        # Rank all neighbors in diffusion map coordinates.
        nbrs = NearestNeighbors(
            # n_neighbors=dm_dims,
            n_neighbors=diff_map.shape[0],
            algorithm='ball_tree'
        ).fit(diff_map)

        nn_distances, nn_indices = nbrs.kneighbors(diff_map)
        nn_distances = nn_distances[:, 1:]
        nn_indices = nn_indices[:, 1:]
        self._nn_dist = nn_distances
        self._nn_idxs = nn_indices
        return nn_distances, nn_indices

    def max_likelihood_pointwise_dimensionality_est(self):
        n_rows = self.n_rows
        nn_dist = self.nn_dist
        nn_idxs = self.nn_idxs 
        most_distinct_points = np.copy(self.most_distinct_points)

        # Maximum Likelihood pointwise dimensionality estimation
        # Hill (1975), Levina and Bickel (2005)
        row_max = np.max(nn_dist, axis=1)
        row_max = row_max.reshape(len(row_max), 1)
        dim_est = np.sum(np.log(row_max / nn_dist), axis=1)

        # Calculate the average dim_est of local neighborhood.
        n_nbrs = min(n_rows // 20, 100)
        nbrs_dim_est = np.average(dim_est[nn_idxs[:, :n_nbrs]], axis=1)
        # nbrs_dim_est = phate_op.graph.interpolate(nbrs_dim_est)

        # Calculate ranking of neighborhood dim_est, from low to high
        temp = nbrs_dim_est.argsort()
        nbrs_dim_est_ranks = np.empty_like(temp)
        nbrs_dim_est_ranks[temp] = np.arange(len(nbrs_dim_est))

        # Make sure that all distinct points are end points (low dim_est), 
        # not branch point (high dim_est)
        low_dim_est_mask = nbrs_dim_est_ranks[most_distinct_points] < n_rows // 2
        most_distinct_points = most_distinct_points[low_dim_est_mask]

        self._most_distinct_points_adjusted = most_distinct_points
        self._n_nbrs = n_nbrs
        self._nbrs_dim_est = nbrs_dim_est
        return n_nbrs, nbrs_dim_est
    
    def diffuse_dirac_for_end_points(self):        
        n_nbrs = self.n_nbrs        
        nbrs_dim_est = self.nbrs_dim_est
        # NOTE: use adjusted distinct points from max_likelihood_pointwise_dimensionality_est
        most_distinct_points = self.most_distinct_points_adjusted

        ##################################
        # DIFFUSING DIRAC FOR END POINTS #
        ##################################
        pnm = self.pnm
        pmn = self.pmn
        opt_t = self.optimal_t
        nn_idxs = self.nn_idxs        
        n_rows = self.n_rows

        branch_points = []
        classes = np.zeros(n_rows, dtype="int32") # NOTE: original was data
        classes_value = np.repeat(-float('inf'), n_rows)
        for end_point_index in np.arange(most_distinct_points.size):
            cur_end_point = most_distinct_points[end_point_index]
                        
            if self.is_landmarked:
                undo_diff = (pmn @ self.diff_op @ pnm)
                diff_op_t = np.linalg.matrix_power(undo_diff, opt_t)
            else:
                diff_op_t = np.linalg.matrix_power(self.diff_op, opt_t)            

            branch_point_dim_est_avg_cache = -float('inf')

            for it in range(self.diffusion_iterations):
                branch_from_end_point = diff_op_t[:, cur_end_point]

                branch_max = np.max(branch_from_end_point)
                branch_min = np.min(branch_from_end_point)
                
                branch_threshold = branch_min + (branch_max - branch_min) * 0.1
                
                deviation_from_branch_threshold = branch_from_end_point - branch_threshold
                deviation_from_branch_threshold[deviation_from_branch_threshold < 0] = float('inf')

                cur_branch_point = deviation_from_branch_threshold.argmin()
                potential_branch_points = np.argpartition(deviation_from_branch_threshold, 20)[:20]
                
                branch_point_dim_est_avg = np.average(nbrs_dim_est[potential_branch_points])
                if (branch_point_dim_est_avg < branch_point_dim_est_avg_cache):
                    break
                branch_point_dim_est_avg_cache = branch_point_dim_est_avg
                                
                if self.is_landmarked:                                
                    undo_diff = (pmn @ self.diff_op @ pnm)
                    diff_op_t = diff_op_t.dot(undo_diff)
                else:
                    diff_op_t = diff_op_t.dot(self.diff_op)

            branch_points.append(cur_branch_point)
            on_branch_mask = diff_op_t[:, cur_end_point] > branch_threshold
            color = diff_op_t[:, cur_end_point]

            on_branch_mask[color < classes_value] = 0

            color[np.logical_not(on_branch_mask)] = -np.max(color)

            classes_value[on_branch_mask] = color[on_branch_mask]
            classes[on_branch_mask] = end_point_index + 1

        #####################
        # REMOVE DUPLICATES #
        #####################
        # We want to remove branch points that are too close together.
        branch_points = np.array(branch_points)
        branch_point_nbrs = nn_idxs[branch_points, :n_nbrs]
        branch_point_pairs_mask = np.isin(branch_point_nbrs, branch_points)
        center_branch_point = branch_points[np.where(branch_point_pairs_mask)[0]]
        neighbor_branch_point = branch_point_nbrs[branch_point_pairs_mask]
        branch_point_pairs = list(zip(center_branch_point, neighbor_branch_point))

        # For each pair of branch_points, keep only the one with higher eigenvalue.
        # (mdb_pairs, by construction, is sorted by decreasing eigenvalue corresponding 
        # to the first point of each pair.)
        points_to_exclude = []
        for pair in branch_point_pairs:
            if pair[0] not in points_to_exclude:
                points_to_exclude.append(pair[1])

        branch_points = np.delete(
            branch_points, 
            np.argwhere(np.isin(branch_points, points_to_exclude))
        )
        self._classes = classes
        self._branch_points = branch_points
        return branch_points

    def assign_branches(self, emb):
        ###################
        # ASSIGN BRANCHES #
        ###################
        dmap = self.dmap
        most_distinct_points = self.most_distinct_points_adjusted

        # Find coordinates between every point and every MDP.
        all_dm_coords = dmap
        if self.is_landmarked:
            all_dm_coords = (self.pmn @ self.dmap @ self.pnm)
        mdp_dm_coords = all_dm_coords[most_distinct_points,:]
        pairwise_dist = sp.spatial.distance.cdist(all_dm_coords, mdp_dm_coords)

        # For every point, rank MDPs by increasing distance.
        s = np.argsort(pairwise_dist, axis=1)
        i = np.arange(pairwise_dist.shape[0]).reshape(-1, 1)
        j = np.arange(pairwise_dist.shape[1])
        
        mdp_ranking = np.empty_like(pairwise_dist, dtype=int)
        mdp_ranking[i, s] = j + 1

        # Assign every point to the branch between its two most highly ranked MDPs.
        mdp_1 = np.argwhere(mdp_ranking==1)[:,1] + 1
        mdp_2 = np.argwhere(mdp_ranking==2)[:,1] + 1
        
        branch_classes = list(zip(mdp_1, mdp_2))
        branch_classes = [str(sorted(branch_class)) for branch_class in branch_classes]
        self._branch_classes = branch_classes 

    def plot_branchs(self, emb):
        most_distinct_points = self.most_distinct_points_adjusted
        branch_points = self.branch_points
        # Plot by class with end points and branch points
        classes = self.classes        
        ax = scprep.plot.scatter2d(emb, c=classes)
        plot_numbers = np.repeat("", emb.shape[0])
        plot_numbers[most_distinct_points] = np.arange(most_distinct_points.shape[0]) + 1
        plot_numbers[branch_points] = "*"
        bbox_props = dict(boxstyle="circle,pad=0.3", fc="w", ec="r", lw=2)
        
        for i, txt in enumerate(plot_numbers):
            ax.annotate(txt, (emb[i][0], emb[i][1]), size=15, bbox=bbox_props)

    def plot_branch_classes(self, emb):
        branch_classes = self.branch_classes
        most_distinct_points = self.most_distinct_points_adjusted

        ax = scprep.plot.scatter2d(emb, c=branch_classes)
        
        plot_numbers = np.repeat("", emb.shape[0])
        plot_numbers[most_distinct_points] = np.arange(most_distinct_points.shape[0]) + 1
        bbox_props = dict(boxstyle="circle,pad=0.3", fc="w", ec="r", lw=2)

        # sys.stdout = open('trash', 'w')
        for i, txt in enumerate(plot_numbers):
            ax.annotate(txt, (emb[i][0], emb[i][1]), size=15, bbox=bbox_props)   
