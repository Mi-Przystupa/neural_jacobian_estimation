from visualservoing.inverse_jacobian import InverseJacobian, MultiPointInverseJacobian, TwoDOFInverseJacobian
from visualservoing.ammortized_local_least_square_uvs import  AmmortizedLocalLeastSquareUVS, ReverseAmmortizedLocalLeastSquareUVS
from visualservoing.broyden_uvs import UncalibratedVisuoServoing
from visualservoing.local_least_square_uvs import  LocalLeastSquareUVS
from visualservoing.state_extractory import StateExtractor
#from visualservoing.rl_vs_wrapper import RL_UVS
from visualservoing.td_uvs import TemporalDifferenceUVS
from visualservoing.black_box_forward_kin import BlackBoxForwardKinematicsUVS
from visualservoing.networks import Network, count_parameters
from visualservoing.random_policy import RandomPolicy
from visualservoing.knn_neural_jacobian import NeuralJacobianKNN
import torch.nn as nn

def PickPolicy(policy_name, gain, num_pts, pts_dim, num_actuators, 
                pose_config='pose_and_origin', activation="relu", num_hiddens=1,
                partial_state="position,angles,velocity,target",
                epochs=30, lengths=[0.3143, 0.1674 + 0.120], beta=1.0, k= 10, l2=0.0 ):
    #TODO: this is really starting to decay...probably should break this up by types of policies and such 
    policy_name = policy_name.lower()
    #TODO a simpler idea is to initialize state_extractor and pass it to the policy creator....
    #TODO probably should use kwargs to pass values for each policy....

    state_extractor = StateExtractor(num_points=num_pts, point_dim=pts_dim, num_angles=num_actuators,partial_state=partial_state)

    inputs = state_extractor.get_partial_state_dimensions() 
    outputs = num_pts * pts_dim * num_actuators

    #could make it an input...figured this is simpler (maybe i'm wrong)
    use_custom_net = "custom" in policy_name 
    if use_custom_net:
        #TODO: for the ammortized networks if using partial info...this will break (hopefully...or there's a bug idk)
        network_inputs = inputs if "blackbox" not in policy_name else num_actuators
        network_outputs = outputs if "blackbox" not in policy_name else num_pts * pts_dim
        if "-fixed" in policy_name:
            policy_name = policy_name.split('-fixed')[0] #remove -fixed part and keep rest of name...assumes policy_name is <existing option>-fixed
            #number of parameters determined from original model we considered
            model = nn.Sequential(*[
                        nn.Linear(network_inputs, 100),
                        nn.ReLU(),
                        nn.Linear(100, 100),
                        nn.ReLU(),
                        nn.Linear(100, network_outputs)
                        ]) #TODo: probably... abetter way to do this

            num_params = count_parameters(model)
            print("Based Network has {} total parameters".format(num_params))
            net = Network.factory("fixed",
                    **{"inputs": network_inputs, "num_params": num_params, 
                                    "num_hiddens": num_hiddens, 
                                    "outputs": network_outputs, "activation": activation})
            actual_params = count_parameters(net)
            print("Network we are using has {} total parameters".format(actual_params))
        else:
            net = Network.factory("mlp", 
                                **{"inputs": network_inputs, "hiddens": [100 for i in range(num_hiddens)], "out": network_outputs, "activation": activation})
        print(net)
    else:
        net = None

    if policy_name == "inversejacobian":
        #with how this controller works the state extract only should pass 1 point
        return InverseJacobian(gain= gain, state_extractor=state_extractor)
    elif policy_name == "multipoint-inversejacobian":
        #assumes 4 points in specific configuration
        return MultiPointInverseJacobian(gain= gain, state_extractor=state_extractor, pts_config=pose_config)
    elif policy_name == "2-dof-inversejacobian":
        return TwoDOFInverseJacobian(gain= gain, L1=lengths[0], L2=lengths[1])
    elif policy_name == "local-uvs":
        return UncalibratedVisuoServoing(gain= gain, n_updates=10, use_broyden=False, state_extractor=state_extractor, num_actuators= num_actuators)
    elif policy_name == "broyden":
        return UncalibratedVisuoServoing(gain= gain, n_updates=10, use_broyden=True, state_extractor=state_extractor, num_actuators= num_actuators)
    elif policy_name == "bfgs":
        #TODO: BFGS is only plausible if number of features = number of actuators else can't be used
        return UncalibratedVisuoServoing(gain= gain, n_updates=10, use_broyden=False, use_BFGS=True, state_extractor=state_extractor, num_actuators= num_actuators)
    elif policy_name == "knn-neuraljacobian":
        #do something
        return NeuralJacobianKNN(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor,
            epochs=epochs, num_actuators= num_actuators, k=k)
    elif policy_name == "knn-neuraljacobian-custom":
        # do something else
        return NeuralJacobianKNN(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor,
            epochs=epochs, num_actuators= num_actuators, k=k, custom_network=net)
    elif policy_name == "multitask-knn-neuraljacobian":
        return NeuralJacobianKNN(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor,
            epochs=epochs, num_actuators= num_actuators, k= k , fit_inverse_relation=True, beta = beta)
    elif policy_name == "multitask-knn-neuraljacobian-custom":
        return NeuralJacobianKNN(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor,
            epochs=epochs, num_actuators= num_actuators, k= k, custom_network=net, fit_inverse_relation=True, beta = beta)
    elif policy_name == "global-locallinear":
        return LocalLeastSquareUVS(gain= gain, k= k, solve_least_square_together=True, num_actuators= num_actuators, state_extractor=state_extractor, use_kd_tree= False)
    elif policy_name == "global-locallinear-kd":
        return LocalLeastSquareUVS(gain= gain, k= k,num_actuators= num_actuators, solve_least_square_together=True, state_extractor=state_extractor, use_kd_tree= True)
    elif policy_name == "global-neuralnetwork":
        return AmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor, epochs=epochs, num_actuators= num_actuators)
    elif policy_name == "global-neuralnetwork-multitask":
        return AmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs,state_extractor=state_extractor, fit_inverse_relation=True, epochs=epochs, num_actuators= num_actuators, beta=beta)
    elif policy_name == "global-neuralnetwork-nullspace":
        return AmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs,state_extractor=state_extractor, fit_null_space=True, epochs=epochs, num_actuators= num_actuators, beta=beta)
    elif policy_name == "global-rbf":
            return AmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor, use_rbf=True,epochs=epochs)
    elif policy_name == "global-rbf-multitask":
            return AmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor,
                                                    use_rbf=True, fit_inverse_relation=True,epochs=epochs, beta=beta)
    elif policy_name == "global-neuralnetwork-custom":
        return AmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor, custom_network = net,epochs=epochs, num_actuators= num_actuators)
    elif policy_name == "global-neuralnetwork-multitask-custom":
        return AmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs,state_extractor=state_extractor, fit_inverse_relation=True, custom_network = net,epochs=epochs, num_actuators= num_actuators, beta=beta)
    elif policy_name == "global-neuralnetwork-nullspace-custom":
        return AmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs,state_extractor=state_extractor, fit_null_space=True, custom_network = net,epochs=epochs, num_actuators= num_actuators, beta=beta)
    elif policy_name == "global-linear":
        return AmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor, use_linear=True,epochs=epochs)
    elif policy_name == "global-linear-multitask":
        return AmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor, fit_inverse_relation=True, use_linear=True,epochs=epochs, beta=beta )
    elif policy_name == "global-reverse-neuralnetwork":
        return ReverseAmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor, epochs=epochs, num_actuators= num_actuators)
    elif policy_name == "global-reverse-neuralnetwork-multitask":
        return ReverseAmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor, epochs=epochs, num_actuators= num_actuators, fit_inverse_relation= True)
    elif policy_name == "global-reverse-neuralnetwork-custom":
        return ReverseAmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor, epochs=epochs, num_actuators= num_actuators, custom_network= net)
    elif policy_name == "global-reverse-neuralnetwork-multitask-custom":
        return ReverseAmmortizedLocalLeastSquareUVS(gain= gain, inputs=inputs, outputs=outputs, state_extractor=state_extractor, epochs=epochs, num_actuators= num_actuators, fit_inverse_relation= True)
    #elif policy_name == "rl_uvs":
    #    return RL_UVS( num_actuators, num_pts, pts_dim, time_steps=5000000, predict_inverse=True)
    elif policy_name == "linear-td":
        return TemporalDifferenceUVS(inputs= inputs, num_feats=num_pts * pts_dim, gain=gain, 
                                    num_actuators=num_actuators, lr=0.1, lam=0.99,
                                    state_extractor = state_extractor, num_sequences=200) 
    elif policy_name == "blackbox-kinematics":
        return  BlackBoxForwardKinematicsUVS(gain = gain, num_actuators= num_actuators, num_feats = num_pts * pts_dim, epochs=epochs, state_extractor=state_extractor, use_linear=False, l2=l2 )
    elif policy_name == "blackbox-rbf":
        return  BlackBoxForwardKinematicsUVS(gain = gain, num_actuators= num_actuators, num_feats = num_pts * pts_dim, epochs=epochs, state_extractor=state_extractor, use_linear=False, use_rbf=True )
    elif policy_name == "blackbox-el":
        return BlackBoxForwardKinematicsUVS(gain = gain, num_actuators= num_actuators, num_feats = num_pts * pts_dim, epochs=epochs, state_extractor=state_extractor, use_el=True )
    elif policy_name == "blackbox-kinematics-custom":
        return  BlackBoxForwardKinematicsUVS(gain = gain, num_actuators= num_actuators, num_feats = num_pts * pts_dim, epochs=epochs, state_extractor=state_extractor, custom_network = net, l2=l2)
    elif policy_name == "blackbox-kinematics-direct":
        #direct in the sense we backpropagate for joint velocity from the objective function instead of finding jacobian
        return  BlackBoxForwardKinematicsUVS(gain = gain, num_actuators= num_actuators, num_feats = num_pts * pts_dim, epochs=epochs, state_extractor=state_extractor, direct_optimize=True )
    elif policy_name == "random":
        return RandomPolicy(gain = gain, num_actuators = num_actuators)





    assert False, "{} is invalid".format(policy_name)
