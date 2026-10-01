"""Offline geometry allowance; physical execution must also check contact force."""
def allowed_tip_contact(bodies, distance, hand_side, enabled):
    return bool(enabled and distance >= -.0002 and set(bodies)=={
        hand_side+'_hand_thumb_2_link', hand_side+'_hand_index_1_link'})
