# openloongretarget

Copyright 2025 国地共建人形机器人创新中心/人形机器人（上海）有限公司, https://www.openloong.net/

欢迎访问 🐉 OpenLoong 开源项目代码仓库！

#### 介绍
青龙重定向项目，使用关键点位姿匹配方法对SMPL格式数据进行重定向,通过优化的retarget算法，实现人体模型与不同机器人URDF结构的快速适配，解决硬件异构难题。

#### 软件架构
SMPL是一套描述人体动作的标准格式和映射算法，用于参数化表达动作和人体体型，其优势在于将体型的不同纳入到动作映射中。通过优化青龙机器人的体型参数 β 使得调整后的人体模型关节位置能够更为贴近青龙机器人。体型参数匹配可以提升后续重定向的动作跟踪精度。

从数据集中收集SMPL格式的动作参数，并使用匹配得到的青龙机器人体型参数 β 将数据集的动作参数θ 映射为机器人各关节目标位置。构造以最小化机器人各关节位姿误差以及连续性误差为优化目标的优化问题，求解最优化问题得到机器人的最优关节角度。


#### 安装教程
创建一个Python3.8版本的虚拟环境,执行下述代码块,your_env_name为自定义的环境名称

```shell
conda create -n your_env_name python=3.8 -y
conda activate your_env_name 
```

进入 `loong_retarget`文件夹后执行下述代码块：


```shell
pip install -e isaacgym/python
pip install -e .
```
#### 使用说明

先下载smpl模型放到`script/retarget/smpl/model/smpl`文件夹下：

1. 下载地址：https://smpl.is.tue.mpg.de/index.html
2. 将模型下载并改名：
```
  |-- smpl
      |-- SMPL_python_v.1.1.0.zip
```
```

  |-- smpl
      |-- SMPL_python_v.1.1.0
        |-- models
            |-- basicmodel_f_lbs_10_207_0_v1.1.0.pkl
            |-- basicmodel_m_lbs_10_207_0_v1.1.0.pkl
            |-- basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
        |-- smpl_webuser
        |-- ...
```
```
  |-- smpl
      |-- SMPL_FEMALE.pkl
      |-- SMPL_MALE.pkl
      |-- SMPL_NEUTRAL.pkl
```

目前项目中只有几个测试demo，其余数据集后续上传至huggingface

主要重定向脚本在`script/retarget/smpl`以及`script/retarget/mink`文件夹下：

1. `grad_fit_openloong_shape.py`用于对青龙机器人进行SMPL体型参数β的匹配。如果没有青龙体型参数文件或者更改了体型匹配权重，那么在进行重定向任务前，应**首先执行一次**该程序。
2. `grad_fit_openloong.py`用于重定向。
3. `loong_retarget_bvh.py`用于动捕设备导出的BVH文件重定向。

可视化脚本在`script/vis`文件夹下：

3. `vis_SMPLVertices.py`用于可视化SMPL匹配效果和动画。
4. `vis_RetargetedResult.py`用于可视化重定向结果中各关节数据。
5. `vis_mujoco.py`用于在mujoco仿真csv的运动数据。
6. `vis_isaacgym.py`用于在isaac gym仿真csv的运动数据。
7. `vis_motion_openloong.py`用于在isaac gym仿真SMPL的pkl的运动数据。

#### 参考开源项目

https://github.com/vchoutas/smplx

https://github.com/ZhengyiLuo/PHC

#### 引用格式

若应用本开源项目中的代码，请以以下格式进行引用：

```JavaScript
@software{Robot2025OpenLoong,
  author = {Humanoid Robot (Shanghai) Co., Ltd},
  title = {{openloongretarget}},
  url = {https://gitee.com/panda_23/openloongretarget},
  year = {2025}
}

@inproceedings{SMPL-X:2019,
    title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
    author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
@article{MANO:SIGGRAPHASIA:2017,
    title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
    author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    volume = {36},
    number = {6},
    series = {245:1--245:17},
    month = nov,
    year = {2017},
    month_numeric = {11}
  }
@article{SMPL:2015,
    author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
    title = {{SMPL}: A Skinned Multi-Person Linear Model},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    month = oct,
    number = {6},
    pages = {248:1--248:16},
    publisher = {ACM},
    volume = {34},
    year = {2015}
}
@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}            
@inproceedings{rempeluo2023tracepace,
    author={Rempe, Davis and Luo, Zhengyi and Peng, Xue Bin and Yuan, Ye and Kitani, Kris and Kreis, Karsten and Fidler, Sanja and Litany, Or},
    title={Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}     

@inproceedings{Luo2022EmbodiedSH,
  title={Embodied Scene-aware Human Pose Estimation},
  author={Zhengyi Luo and Shun Iwase and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{Luo2021DynamicsRegulatedKP,
  title={Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation},
  author={Zhengyi Luo and Ryo Hachiuma and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

```

#### 联系方式

欢迎各位开发者参与本代码库的优化与提高！

您可以对现有内容进行意见评价、问题反馈、贡献您的原创内容等，对本代码的任何问题及意见，请联系<open@openloong.org.cn>