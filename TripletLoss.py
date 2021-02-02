class TripletLoss(nn.Module):
    def __init__(self, margin=.2, negative='max'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.negative = negative

    def forward(self, anchor, positive):
        dists = torch.cdist(anchor, positive)

        p_dists = torch.diag(dists)
        p_dist = p_dists.unsqueeze(1).expand_as(dists)

        cost = (p_dist-dists) + self.margin
        cost = cost.clamp(min=0).fill_diagonal_(0)
        # print(cost)
        if self.negative == 'max':
            cost = torch.max(cost, keepdim=True, dim=1)[0]
            # print(cost)
        elif self.negative == 'random':
            pass
        elif self.negative == 'all':
            pass
        else:
            raise ValueError("error")

        return cost[cost > 0].mean()
