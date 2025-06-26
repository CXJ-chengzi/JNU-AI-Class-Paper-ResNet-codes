from graphviz import Digraph

def create_resnet18_graph():
    """创建高清晰度的ResNet-18流程图"""
    dot = Digraph(comment='ResNet-18 Architecture', format='png')
    # 核心优化：提高dpi（像素密度）并调整画布尺寸
    dot.attr(rankdir='TB', size='16,24', dpi='600')  # dpi：600
    dot.attr('node', shape='box', style='filled', color='lightblue', 
             fontname='SimHei', fontsize='14')  # 增大字体
    
    # 输入层
    dot.node('input', '输入图像\n3×224×224')
    
    # 初始卷积+池化（优化节点间距）
    dot.node('conv1', '卷积层1\n7×7, 64, stride=2\n64×112×112')
    dot.node('bn1', '批量归一化')
    dot.node('relu1', 'ReLU激活')
    dot.node('maxpool', '最大池化\n3×3, stride=2\n64×56×56')
    
    # 连接第一阶段
    dot.edge('input', 'conv1')
    dot.edge('conv1', 'bn1')
    dot.edge('bn1', 'relu1')
    dot.edge('relu1', 'maxpool')
    
    # 残差块组（优化层级间距）
    prev_output = 'maxpool'
    dot.attr(ranksep='2.0')  # 增大垂直间距
    
    # Layer 1: 2个残差块
    for i in range(2):
        block_id = f'layer1_{i+1}'
        add_basic_block(dot, block_id, 64, 64, 1, prev_output)
        prev_output = f'{block_id}_out'
    
    # Layer 2: 2个残差块（通道128）
    for i in range(2):
        block_id = f'layer2_{i+1}'
        stride = 2 if i == 0 else 1
        in_ch = 64 if i == 0 else 128
        add_basic_block(dot, block_id, in_ch, 128, stride, prev_output)
        prev_output = f'{block_id}_out'
    
    # Layer 3: 2个残差块（通道256）
    for i in range(2):
        block_id = f'layer3_{i+1}'
        stride = 2 if i == 0 else 1
        in_ch = 128 if i == 0 else 256
        add_basic_block(dot, block_id, in_ch, 256, stride, prev_output)
        prev_output = f'{block_id}_out'
    
    # Layer 4: 2个残差块（通道512）
    for i in range(2):
        block_id = f'layer4_{i+1}'
        stride = 2 if i == 0 else 1
        in_ch = 256 if i == 0 else 512
        add_basic_block(dot, block_id, in_ch, 512, stride, prev_output)
        prev_output = f'{block_id}_out'
    
    # 最终层
    dot.node('avgpool', '全局平均池化\n512×7×7 → 512×1×1')
    dot.node('fc', '全连接层\n512 → 1000')
    dot.node('output', '输出\n1000维')
    
    # 连接最终层
    dot.edge(prev_output, 'avgpool')
    dot.edge('avgpool', 'fc')
    dot.edge('fc', 'output')
    
    return dot

def add_basic_block(dot, block_id, in_channels, out_channels, stride, prev_output):
    """添加残差块（优化节点尺寸和文字换行）"""
    # 主路径节点（优化文字显示，增加换行）
    conv1_id = f'{block_id}_conv1'
    dot.node(conv1_id, f'卷积\n3×3, {out_channels}\nstride={stride}\n{out_channels}×H/stride×W/stride')
    dot.node(f'{block_id}_bn1', '批量归一化')
    dot.node(f'{block_id}_relu1', 'ReLU激活')
    dot.node(f'{block_id}_conv2', f'卷积\n3×3, {out_channels}\nstride=1\n{out_channels}×H×W')
    dot.node(f'{block_id}_bn2', '批量归一化')
    
    # 捷径连接
    if stride != 1 or in_channels != out_channels:
        shortcut_id = f'{block_id}_shortcut'
        dot.node(shortcut_id, f'1×1卷积\n{out_channels}, stride={stride}\n通道匹配')
        dot.edge(prev_output, shortcut_id)
        dot.edge(shortcut_id, f'{block_id}_bn2')
        shortcut_input = shortcut_id
    else:
        dot.edge(prev_output, f'{block_id}_bn2', style='dashed')
        shortcut_input = prev_output
    
    # 连接主路径
    dot.edge(prev_output, conv1_id)
    dot.edge(conv1_id, f'{block_id}_bn1')
    dot.edge(f'{block_id}_bn1', f'{block_id}_relu1')
    dot.edge(f'{block_id}_relu1', f'{block_id}_conv2')
    dot.edge(f'{block_id}_conv2', f'{block_id}_bn2')
    
    # 输出节点
    relu_out_id = f'{block_id}_out'
    dot.node(relu_out_id, 'ReLU激活')
    dot.edge(f'{block_id}_bn2', relu_out_id)

if __name__ == "__main__":
    dot = create_resnet18_graph()
    # 生成高清晰度文件（cleanup=True自动删除中间文件）
    dot.render('resnet18_highres', view=True, cleanup=True)
    print("已生成高清晰度流程图：resnet18_highres.png")