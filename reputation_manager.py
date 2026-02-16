import config

class ReputationManager:
    def __init__(self):
        self.reputations = {} # Dictionary mapping client_id -> reputation_score (0.0 to 1.0)
        self.alpha = config.REPUTATION_ALPHA
        self.ban_threshold = config.REPUTATION_BAN_THRESHOLD
        self.grace_period = getattr(config, 'REPUTATION_GRACE_PERIOD', 0)

    def update_reputations(self, client_ids, current_scores, current_round=None):
        """
        Updates reputation scores for the given clients based on their current round performance.
        
        Args:
            client_ids (list): List of client IDs (strings or ints) participating in this round.
            current_scores (list/tensor): List of raw scores from the aggregator for these clients.
            current_round (int): The current round number. Used for Grace Period logic.
            
        Returns:
            list: A list of client_ids using the Zero-Tolerance policy (Reputation = 0.0).
        """
        banned_in_current_round = []
        
        # --- Grace Period Check ---
        if current_round is not None and current_round <= self.grace_period:
            # During grace period, we update metrics if we wanted to track history, 
            # but for this strict system, we simply skip the BANNING logic.
            # We can still apply the EMA update to let good clients build trust, 
            # OR we can skip updates entirely to avoid punishing early variance.
            # DECISION: Skip updates entirely to be safe.
            # print(f"    [ReputationManager] Grace Period (Round {current_round} <= {self.grace_period}). Skipping updates.")
            return []

        # Ensure inputs are aligned
        if len(client_ids) != len(current_scores):
            print(f"    [ReputationManager] Error: Mismatch in clients ({len(client_ids)}) and scores ({len(current_scores)})")
            return []

        for i, client_id in enumerate(client_ids):
            score = float(current_scores[i])
            
            # Initialize if not present
            if client_id not in self.reputations:
                self.reputations[client_id] = 0.5
            
            old_rep = self.reputations[client_id]
            
            # --- Already banned? Stay banned. ---
            if old_rep == 0.0:
                new_rep = 0.0
            else:
                # --- Step 1: Apply EMA Update FIRST ---
                new_rep = (self.alpha * old_rep) + ((1 - self.alpha) * score)
                new_rep = max(0.0, min(1.0, new_rep))
                
                # --- Step 2: THEN check if EMA reputation fell below threshold ---
                if new_rep < self.ban_threshold:
                    new_rep = 0.0
                    print(f"    [ReputationManager] BANNING Client {client_id} (EMA Rep: {(self.alpha * old_rep) + ((1 - self.alpha) * score):.4f} < {self.ban_threshold})")

            self.reputations[client_id] = new_rep
            
            if new_rep == 0.0:
                banned_in_current_round.append(client_id)

        return banned_in_current_round

    def get_reputation(self, client_id):
        return self.reputations.get(client_id, 0.5)

    def get_banned_clients(self):
        """Returns a list of all currently banned clients."""
        return [cid for cid, rep in self.reputations.items() if rep == 0.0]

    def get_weights(self, client_ids):
        """
        Returns normalized weights for the given list of clients based on their reputation.
        Useful if we want to weight the aggregation by reputation (though Aegis does its own scoring).
        """
        reps = [self.get_reputation(cid) for cid in client_ids]
        total_rep = sum(reps)
        
        if total_rep == 0:
            # Avoid division by zero, return uniform weights
            return [1.0 / len(client_ids)] * len(client_ids)
            
        return [r / total_rep for r in reps]
