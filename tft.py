
import matplotlib.pyplot as plt 
RAGEBLADE_AS = 0.1
RAGEBLADE_SCALE = 0.05
RAGEBLADE_AP = 10
KAISA_SPELL_AS = [0, 0.45, 0.6, 0.75]
KAISA_SPELL_TIME = 10
KAISA_PLASMA_COUNT = 2
KAISA_PLASMA_DAMAGE = [0 ,140, 210, 350]
SHOJIN_AP = 15
BLUEBUFF_AP = 15
DCAP_AP =  65
SHOJIN_MANA = 20

class Champion:
    
    def __init__(self, attack_damage, attack_speed, mana_cost, starting_mana, star_level, items, sg_level, heart_ap):
        self.base_attack_damage = attack_damage
        self.base_attack_speed = attack_speed
        self.mana_cost = mana_cost
        self.starting_mana = starting_mana
        if(items['Shojin']):
            self.starting_mana += 15
        if(items['BlueBuff']):
            self.mana_cost -= 10
            self.starting_mana +=60
        self.cur_mana = self.starting_mana
        self.managen = 10
        self.items = items
        self.has_rageblade = items['Rageblade']
        self.attack_damage = attack_damage
        self.attack_speed = attack_speed 
        self.star_level = star_level
        self.rageblade_stacks = 0
        self.base_ap = 100
        self.cur_ap = self.calc_ap()
        self.star_guardian_level = sg_level
        self.heart_ap = heart_ap

    def calc_ap(self):
        ap = self.base_ap
        if(self.items['Rageblade']):
            ap += self.base_ap * RAGEBLADE_AP/100
        if(self.items['Shojin']):
            ap += self.base_ap * SHOJIN_AP/100
        if(self.items['BlueBuff']):
            ap += self.base_ap * BLUEBUFF_AP/100
        if(self.items['Rabbadons']):
            ap += self.base_ap * DCAP_AP/100
        
        return ap

    def run_base(self, num_autos):
        cum_damage = [0]
        time = [0]
        for i in range((num_autos)):
            auto_time = 1/self.attack_speed
            time.append(time[i] + auto_time)
            cum_damage.append(self.attack_damage + cum_damage[i])
        return time, cum_damage 

    def update_rageblade(self):
        if(self.has_rageblade):
            self.rageblade_stacks += 1
            self.attack_speed = self.base_attack_speed * (1+RAGEBLADE_SCALE * self.rageblade_stacks)
        if(self.attack_speed > 5):
            self.attack_speed = 5

    def update_as(self, attack_speed_scale, as_stacks):
        
        self.attack_speed = self.base_attack_speed +(self.base_attack_speed* attack_speed_scale * as_stacks)
        self.attack_speed = self.attack_speed * (1+RAGEBLADE_SCALE * self.rageblade_stacks)
        if(self.attack_speed > 5):
            self.attack_speed = 5

    def run_rageblade(self, num_autos):
        cum_damage = [0]
        time = [0]
        if(self.has_rageblade):
            self.attack_speed = self.base_attack_speed * (1+RAGEBLADE_AS)
        for i in range((num_autos)):
            auto_time = 1/self.attack_speed
            self.update_rageblade()
            time.append(time[i] + auto_time)

            cum_damage.append(self.attack_damage + cum_damage[i])
        return time, cum_damage 
    def get_sg_ratio(self):
        if(self.star_guardian_level<3):
            return 1
        if(self.star_guardian_level<5):
            return 1.4
        if(self.star_guardian_level<7):
            return 1.7
        if(self.star_guardian_level<9):
            return 2.2
        return 3.0
    def update_mana(self, auto_num):
        
            
        self.cur_mana = self.cur_mana + self.managen*self.get_sg_ratio()
        if(self.items['Shojin'] and auto_num % 3 == 2):
            self.cur_mana = self.cur_mana + SHOJIN_MANA*self.get_sg_ratio()
        
        if(self.cur_mana >= self.mana_cost):
            self.cur_mana =0#-= self.mana_cost
            self.cur_ap += self.heart_ap
            if(self.items['BlueBuff']):
                self.cur_mana += 0#10*self.get_sg_ratio()
            return True 
        return False 

    def end_spell(self, cast_times, time):
        if(len(cast_times) < 1):
            return False 
        if (time >= cast_times[0] + KAISA_SPELL_TIME):
            cast_times.pop(0)
            return True 
        return False

    def run_kaisa(self, run_time):
        
        cum_damage = [0]
        time = [0]
        cast_times = []
        if(self.has_rageblade):
            
            self.attack_speed = self.base_attack_speed * (1+RAGEBLADE_AS)
        num_autos = 999
        for i in range((num_autos)):
            auto_time = 1/self.attack_speed
            self.update_rageblade()
            cur_time = time[i] + auto_time
            
            time.append(cur_time)
            has_cast = self.update_mana(i)
            if(has_cast):
                cast_times.append(cur_time)
            spell_end = self.end_spell(cast_times, cur_time)
            self.update_as(KAISA_SPELL_AS[self.star_level], len(cast_times))
            
            #print(str(cur_time) + ',' + 
            #        str(self.attack_speed) + ',' + 
            #        str(self.cur_mana)+','  + str(cast_times) )
            cur_damage = self.attack_damage
            if(i%3 == 2):
                cur_damage += KAISA_PLASMA_DAMAGE[self.star_level]*self.cur_ap/100
            cum_damage.append(cur_damage + cum_damage[i])
            if(cur_time > run_time):
                break
        return time, cum_damage 

    
             

        


def main():
    kaisa_health = 650
    kaisa_ad = 40
    kaisa_as = 0.96
    kaisa_mana = 60
    kaisa_startmana = 0
    kaisa_starlevel = 1
    sg_level = 3
    runtime = 30
    items = {'Rageblade': False, 
                'Shojin': False,
                'BlueBuff': False,
                'Rabbadons': True}
    for heart_ap in [0, 4, 8, 12]:
    
        for kaisa_as in [0.8, 0.96]:
            fig, axes = plt.subplots(3,5, figsize=(30,60))
            for sg_level in [0,3,5,7,9]:
                for kaisa_starlevel in [1, 2,3]:
                    items['BlueBuff'] = False
                    kaisa = Champion(kaisa_ad, kaisa_as, kaisa_mana, kaisa_startmana, kaisa_starlevel, items, sg_level, heart_ap)
                    
                    t0, d0 = kaisa.run_kaisa(runtime) 
                    items['Rageblade'] = True
                    kaisa = Champion(kaisa_ad, kaisa_as, kaisa_mana, kaisa_startmana, kaisa_starlevel, items, sg_level, heart_ap)
                    
                    t1, d1 = kaisa.run_kaisa(runtime) 
                    items['Rageblade'] = False
                    items['Shojin'] = True
                    kaisa = Champion(kaisa_ad, kaisa_as, kaisa_mana, kaisa_startmana, kaisa_starlevel, items, sg_level,heart_ap)
                    t2, d2 = kaisa.run_kaisa(runtime) 

                    items['Shojin'] = False
                    items['BlueBuff'] = True
                    kaisa = Champion(kaisa_ad, kaisa_as, kaisa_mana, kaisa_startmana, kaisa_starlevel, items, sg_level, heart_ap)
                    t3, d3 = kaisa.run_kaisa(runtime) 

                    ax = axes[kaisa_starlevel-1,int((sg_level-1)/2)]
                    title_str = ('Kaisa at Star:' + str(kaisa_starlevel) + 
                                    ", Starting AS:" + str(kaisa_as) +
                                    ", Starguardians:" + str(sg_level))
                    ax.set_title(title_str)
                    ax.plot(t0, d0, label='No Items', color='Black')
                    ax.plot(t1, d1, label='Rageblade', color='Red')
                    ax.plot(t2, d2, label='Shojin', color='Green')
                    ax.plot(t3, d3, label='BlueBuff', color='Blue')
                    ax.set_ylabel('Damage')
                    ax.set_ylim(0,50000)
                ax.set_xlabel('Time (s)')
            
            ax.legend()
            suptitle_str = 'AS'+str(kaisa_as)+'_Heart'+str(heart_ap)
            fig.suptitle(suptitle_str)
            fig.tight_layout()
            plt.show()
            print(
                'saving fig: ' + suptitle_str
            )
            #plt.savefig(suptitle_str +'rabadon.png')


if __name__ == '__main__':
    main()