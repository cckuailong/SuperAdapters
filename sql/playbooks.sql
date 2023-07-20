/*
 Navicat Premium Data Transfer

 Source Server         : mamp_mysql
 Source Server Type    : MySQL
 Source Server Version : 50734
 Source Host           : localhost:3306
 Source Schema         : llm

 Target Server Type    : MySQL
 Target Server Version : 50734
 File Encoding         : 65001

 Date: 20/07/2023 17:04:24
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for playbooks
-- ----------------------------
DROP TABLE IF EXISTS `playbooks`;
CREATE TABLE `playbooks` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `payload_uuid` varchar(64) DEFAULT NULL,
  `type` varchar(16) DEFAULT NULL,
  `instruction` text,
  `input` longtext,
  `output` longtext,
  `iteration` varchar(32) DEFAULT NULL,
  `insert_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=131992 DEFAULT CHARSET=utf8mb4;

SET FOREIGN_KEY_CHECKS = 1;
