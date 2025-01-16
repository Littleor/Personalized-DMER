import numpy as np
import torch


def ccc(x, y):
    """Concordance Correlation Coefficient"""
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    sxy = torch.sum((x - x_mean) * (y - y_mean)) / x.shape[0]
    rhoc = (
        2
        * sxy
        / (
            torch.var(x, unbiased=False)
            + torch.var(y, unbiased=False)
            + (x_mean - y_mean) ** 2
        )
    )

    return rhoc


def pcc(p, y):
    """
    Compute the Pearson correlation coefficient
    :param p: the predicted values
    :param y: the ground truth values
    :return: the Pearson correlation coefficient
    """
    p = p - torch.mean(p)
    y = y - torch.mean(y)
    epsilon = 1e-8
    pcc = torch.sum(p * y) / (torch.sqrt(torch.sum(p**2)) * torch.sqrt(torch.sum(y**2)) + epsilon)
    return pcc


def rmse(p, y):
    """
    Compute the root mean square error
    :param p: the predicted values
    :param y: the ground truth values
    :return: the root mean square error
    """
    rmse = torch.sqrt(torch.mean((p - y) ** 2))
    return rmse


def get_score(p: torch.Tensor, y: torch.Tensor, return_tensor: bool = False):
    return_value =  {"CCC": ccc(p, y), "PCC": pcc(p, y), "RMSE": rmse(p, y)}
    if not return_tensor:
        for k, v in return_value.items():
            return_value[k] = v.item()
    return return_value


if __name__ == "__main__":
    p = torch.randn(60)
    y = torch.randn(60)
    print(
        "CCC",
        ccc(p, y),
    )
    print("PCC", pcc(p, y), pcc(y, p))
    print("RMSE", rmse(p, y))
    print(get_score(p, y))
    print(get_score(y, p))
    
    print("---")
    print(get_score(
        p.unsqueeze(0),
        y.unsqueeze(0)
    ))
    print(get_score(
        p.unsqueeze(0).repeat(5, 1),
        y.unsqueeze(0).repeat(5, 1)
    ))
