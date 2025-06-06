from gr00t.model.action_head.flow_matching_action_head import CategorySpecificMLP
import torch


def test_category_specific_mlp_basic():
    """基础功能测试：验证输入输出形状及类别参数隔离"""
    # 初始化参数
    num_categories = 3
    input_dim = 5
    hidden_dim = 8
    output_dim = 2
    batch_size = 4
    seq_len = 2

    # 创建模型
    model = CategorySpecificMLP(num_categories, input_dim, hidden_dim, output_dim)

    # 生成测试数据
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, input_dim)

    cat_ids = torch.tensor([0, 1, 2, 0])  # 包含不同类别ID

    # 前向传播
    output = model(x, cat_ids)

    # 验证输出形状
    assert output.shape == (batch_size, seq_len, output_dim), "输出形状错误"
    print("x.shape ", x.shape)
    print("output.shape ", output.shape)
    # 验证不同类别产生不同输出
    same_cat_output = model(x[[0, 3]], cat_ids[[0, 3]])
    assert torch.allclose(output[0], same_cat_output[0], atol=1e-6), "相同类别应产生相似输出"
    assert not torch.allclose(output[0], output[1], atol=1e-4), "不同类别应产生不同输出"


if __name__ == "__main__":
    test_category_specific_mlp_basic()
